from pathlib import Path

import tfs
from cpymad.madx import Madx
from pymadng import MAD

from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.model_creator import create_instance_and_model
from omc3.optics_measurements.constants import RDT_FOLDER
from rdt_constants import (
    ANALYSIS_DIR,
    DATA_DIR,
    FREQ_OUT_DIR,
    CURRENT_DIR,
    ACC_MODELS,
    NORMAL_SEXTUPOLE_RDTS,
    SKEW_SEXTUPOLE_RDTS,
)
ANALYSIS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
FREQ_OUT_DIR.mkdir(exist_ok=True)

def get_file_suffix(beam: int, nturns: int) -> str:
    """Return the file suffix for the test files based on beam and order."""
    assert beam in [1, 2], "Beam must be 1 or 2"
    return f"b{beam}_{nturns}t"

def get_model_dir(beam: int) -> Path:
    """Return the model directory for the given test parameters."""
    return CURRENT_DIR / f"model_b{beam}"

def get_analysis_dir(beam: int, nturns: int, index: int) -> Path:
    """Return the analysis directory for the given test parameters."""
    tbt_path = get_tbt_path(beam, nturns, index)
    analysis_directory = ANALYSIS_DIR / tbt_path.stem 
    analysis_directory.mkdir(exist_ok=True)
    return analysis_directory

def get_tfs_path(beam: int, nturns: int) -> Path:
    """Return the name of the TFS file for the given test parameters."""
    return DATA_DIR / f"{get_file_suffix(beam, nturns)}.tfs.bz2"

def get_tbt_path(beam: int, nturns: int, index: int) -> Path:
    """Return the name of the TBT file for the given test parameters."""
    suffix = get_file_suffix(beam, nturns) + (f"_{index}" if index != -1 else "_zero_noise")
    return DATA_DIR / f"tbt_{suffix}.sdds"


MADX_FILENAME = "job.create_model_nominal.madx"

def create_model_dir(beam: int) -> None:
    model_dir = get_model_dir(beam)
    create_instance_and_model(
        accel="lhc",
        fetch="afs",
        type="nominal",
        beam=beam,
        year="2024",
        driven_excitation="acd",
        energy=6800.0,
        nat_tunes=[0.28, 0.31],
        drv_tunes=[0.28, 0.31],
        modifiers=[ACC_MODELS / "operation/optics/R2024aRP_A30cmC30cmA10mL200cm.madx"],
        outputdir=model_dir,
    )
    with open(model_dir / MADX_FILENAME, "r") as f:
        lines = f.readlines()
    
    # Make the sequence as beam 1 or 2
    make_madx_seq(model_dir, lines, beam)

    # Update the model by using beam 1 or 2.
    update_model_with_ng(beam)

    # If beam 2, now we need to make the sequence as beam 4 for tracking
    if beam == 2:
        make_madx_seq(model_dir, lines, beam, beam4=True)

def make_madx_seq(
    model_dir: Path, lines: list[str], beam: int, beam4: bool = False
) -> None:
    with Madx(stdout=False) as madx:
        madx.chdir(str(model_dir))
        for i, line in enumerate(lines):
            if beam4:
                if "define_nominal_beams" in line:
                    madx.input(
                        "beam, sequence=LHCB2, particle=proton, energy=450, kbunch=1, npart=1.15E11, bv=1;\n"
                    )
                    continue
                elif "acc-models-lhc/lhc.seq" in line:
                    line = line.replace(
                            "acc-models-lhc/lhc.seq", "acc-models-lhc/lhcb4.seq"
                        )
            if i < 32:
                madx.input(line)
        madx.input(
            f"""
set, format= "-.16e";
save, sequence=lhcb{beam}, file="lhcb{beam}_saved.seq", noexpr=false;
        """)

def update_model_with_ng(beam: int) -> None:
    model_dir = get_model_dir(beam)
    with MAD() as mad:
        seq_dir = -1 if beam == 2 else 1
        initialise_model(mad, beam, seq_dir=seq_dir)
        # Create twiss_elements and twiss_ac and twiss data tables in model folder
        mad.send(f"""
hnams = {{
    "name", "type", "title", "origin", "date", "time", "refcol", "direction", 
    "observe", "energy", "deltap", "length", "q1", "q2", "q3", "dq1", "dq2", 
    "dq3", "alfap", "etap", "gammatr"
}}        
cols = {{
    'name', 'kind','s','betx','alfx','bety','alfy', 'mu1' ,'mu2', 'r11', 'r12',
    'r21', 'r22', 'x','y','dx','dpx','dy','dpy'
}}
str_cols = py:recv()
cols = MAD.utility.tblcat(cols, str_cols)
if {beam} == 1 then  -- Cycle the sequence to the correct location 
    MADX.lhcb1:cycle("MSIA.EXIT.B1")
end
-- Calculate the twiss parameters with coupling and observe the BPMs
! Coupling needs to be true to calculate Edwards-Teng parameters and R matrix
twiss_elements = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true}} 
-- Select everything
twiss_elements:select(nil, \ -> true)
-- Deselect the drifts
twiss_elements:deselect{{pattern="drift"}}
""").send(strength_cols)
        # MAD.gphys.melmcol(twiss_elements, str_cols)
        add_strengths_to_twiss(mad, "twiss_elements")
        mad.send(
            # True below is to make sure only selected rows are written
            f"""twiss_elements:write("{model_dir / 'twiss_elements.dat'}", cols, hnams, true)"""
        )
        observe_BPMs(mad, beam)
        mad.send(f"""
twiss_ac   = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1}}
twiss_data = twiss {{sequence=MADX.lhcb{beam}, mapdef=4, coupling=true, observe=1}}
        """)
        add_strengths_to_twiss(mad, "twiss_ac")
        add_strengths_to_twiss(mad, "twiss_data")
        mad.send(f"""
twiss_ac:write("{model_dir / 'twiss_ac.dat'}", cols, hnams)
twiss_data:write("{model_dir / 'twiss.dat'}", cols, hnams)
print("Replaced twiss data tables")
py:send(1)
""")
        assert mad.receive() == 1, "Error in updating model with new optics"

    # Read the twiss data tables and then convert all the headers to uppercase and column names to uppercase
    export_tfs_to_madx(model_dir / "twiss_ac.dat")
    export_tfs_to_madx(model_dir / "twiss_elements.dat")
    export_tfs_to_madx(model_dir / "twiss.dat")

def observe_BPMs(mad: MAD, beam: int) -> None:
    mad.send(f"""
local observed in MAD.element.flags
MADX.lhcb{beam}:deselect(observed)
MADX.lhcb{beam}:  select(observed, {{pattern="BPM"}})
    """)


def export_tfs_to_madx(tfs_file: Path) -> None:
    tfs_df = tfs.read(tfs_file)
    tfs_df = convert_tfs_to_madx(tfs_df)
    tfs.write(tfs_file, tfs_df, save_index="NAME")


def convert_tfs_to_madx(tfs_df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    # First convert all the headers to uppercase and column names to uppercase
    tfs_df.columns = tfs_df.columns.str.upper()
    tfs_df.headers = {key.upper(): value for key, value in tfs_df.headers.items()}

    # Change the columns mu1 and mu2 to mux and muy
    tfs_df.rename(columns={"MU1": "MUX", "MU2": "MUY"}, inplace=True)

    # Change all the drift numbers (the # in DRIFT_#) are consecutive and start from 0
    drifts = tfs_df[tfs_df["KIND"] == "drift"]
    replace_column = [f"DRIFT_{i}" for i in range(len(drifts))]
    tfs_df["NAME"] = tfs_df["NAME"].replace(drifts["NAME"].to_list(), replace_column)

    # Remove all rows that has a "vkicker" or "hkicker" in the KIND column (not seen in MADX)
    tfs_df = tfs_df[~tfs_df["KIND"].str.contains("vkicker|hkicker")]

    tfs_df.set_index("NAME", inplace=True)

    # Remove the rows with "$start" and "$end" in the NAME column
    tfs_df = tfs_df.filter(regex=r"^(?!\$start|\$end).*$", axis="index")
    return tfs_df


strength_cols = ["k1l", "k2l", "k3l", "k4l", "k5l", "k1sl", "k2sl", "k3sl", "k4sl", "k5sl"]


def add_strengths_to_twiss(mad: MAD, mtable_name: str) -> None:
    mad.send(f"""
strength_cols = py:recv()
MAD.gphys.melmcol({mtable_name}, strength_cols)
    """).send(strength_cols)

def initialise_model(
    mad: MAD,
    beam: int,
    seq_dir: int = 1,
) -> None:
    assert beam in [1, 2] and isinstance(beam, int), "Beam must be 1 or 2"
    model_dir = get_model_dir(beam)
    mad.MADX.load(
        f"'{model_dir/f'lhcb{beam}_saved.seq'}'",
        f"'{model_dir/f'lhcb{beam}_saved.mad'}'",
    )
    mad.send(f"""
lhc_beam = beam {{particle="proton", energy=450}}
MADX.lhcb{beam}.beam = lhc_beam
MADX.lhcb{beam}.dir = {seq_dir}
print("Initialising model with beam:", {beam}, "dir:", MADX.lhcb{beam}.dir)
    """)
    if seq_dir == 1 and beam == 1:
        mad.send("""MADX.lhcb1:cycle("MSIA.EXIT.B1")""")

    ensure_bends_are_on(mad, beam)
    match_tunes(mad, beam)
    add_magnet_strengths(mad)


SEXTUPOLE_STRENGTH = 3e-5
OCTUPOLE_STRENGTH = 3e-3

MAGNET_STRENGTHS = {
    "s" : SEXTUPOLE_STRENGTH,
    # "o" : OCTUPOLE_STRENGTH
}

def add_magnet_strengths(mad: MAD) -> None:
    for s_or_o, strength in MAGNET_STRENGTHS.items():
        for skew in [""]:#  ["", "s"]:
            mad.send(f"""
        MADX.kc{s_or_o}{skew}x3_r1 = {strength:+.16e};
        MADX.kc{s_or_o}{skew}x3_l5 = {-strength:+.16e};
        print("magnets kc{s_or_o}{skew}x3 set to {strength:.16e}")
        """)


def ensure_bends_are_on(mad: MAD, beam: int) -> None:
    mad.send(f"""
for i, element in MADX.lhcb{beam}:iter() do
    if (element.kind == "sbend" or element.kind == "rbend") and (element.angle ~= 0 and element.k0 == 0) then
        element.k0 = \s->s.angle/s.l -- restore deferred expression
    end
end
    """)


def match_tunes(mad: MAD, beam: int) -> None:
    mad.send(rf"""
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4}}
print("Initial tunes: ", tbl.q1, tbl.q2)
match {{
  command := twiss {{sequence=MADX.lhcb{beam}, mapdef=4}},
  variables = {{ rtol=1e-6, -- 1 ppm
    {{ var = 'MADX.dqx_b{beam}_op', name='dQx.b{beam}_op' }},
    {{ var = 'MADX.dqy_b{beam}_op', name='dQy.b{beam}_op' }},
  }},
  equalities = {{
    {{ expr = \t -> math.abs(t.q1)-62.28, name='q1' }},
    {{ expr = \t -> math.abs(t.q2)-60.31, name='q2' }},
  }},
  objective = {{ fmin=1e-7 }},
}}
local tbl = twiss {{sequence=MADX.lhcb{beam}, mapdef=4}}
print("Final tunes: ", tbl.q1, tbl.q2)
py:send("match complete")
""")
    assert mad.receive() == "match complete", "Error in matching tunes"

def run_harpy(beam: int, tbt_file: Path, clean=False) -> None:
    """Run Harpy for the given test parameters."""
    hole_in_one_entrypoint(
        harpy=True,
        files=[tbt_file],
        outputdir=FREQ_OUT_DIR,
        to_write=["lin", "full_spectra"],
        opposite_direction=beam == 2,
        tunes=[0.28, 0.31, 0.0],
        natdeltas=[0.0, -0.0, 0.0],
        turn_bits=18,
        clean=clean,
    )
    for file in FREQ_OUT_DIR.glob("*.ini"):
        file.unlink() # Clean up these really annoying files


def run_tracking(beam: int, nturns: int, kick_amp: float = 1e-2) -> tfs.TfsDataFrame:
    with MAD() as mad:
        initialise_model(mad, beam)
        observe_BPMs(mad, beam)
        # Octupolar resonances are harder to observe with only 1000 turns 
        # so we need to increase the kick amplitude for order 3
        mad.send(f"""
local t0 = os.clock()
local kick_amp = py:recv()
local sqrt_betx = math.sqrt(115.9267454)
local sqrt_bety = math.sqrt(50.09893796)
local X0 = {{x=kick_amp / sqrt_betx, y=-kick_amp/sqrt_bety, px=0, py=0, t=0, pt=0}}
print("Running MAD-NG track with kick amplitude in x: ", X0.x, " and y: ", X0.y)
mtbl = track {{sequence=MADX.lhcb{beam}, nturn={nturns}, X0=X0}}
print("NG Runtime: ", os.clock() - t0)
        """).send(kick_amp)
        df = mad.mtbl.to_df(columns=["name", "x", "y", "eidx", "turn", "id"])
    return df



def get_rdt_names() -> tuple[str]:
    """Return the all the RDTs."""
    return NORMAL_SEXTUPOLE_RDTS + SKEW_SEXTUPOLE_RDTS

def get_output_dir(tbt_name: str, output_dir: Path = None) -> Path:
    """Return the output directory for the given TBT, and create it if it does not exist."""
    if output_dir is None:
        output_dir = ANALYSIS_DIR / f"{tbt_name.split('.')[0]}"
        output_dir.mkdir(exist_ok=True)
    return output_dir


def filter_out_BPM_near_IPs(df: tfs.TfsDataFrame) -> tfs.TfsDataFrame:
    """Filter the DataFrame to include only BPMs."""
    return df.filter(regex=r"^BPM\.[1-9][0-9].", axis="index")

def get_tunes(output_dir: Path) -> list[float]:
    optics_file = output_dir / "beta_amplitude_x.tfs"
    headers = tfs.reader.read_headers(optics_file)
    tunes = [headers["Q1"], headers["Q2"]]
    return tunes


def get_rdt_type(rdt: str) -> tuple[str, str]:
    rdt_as_list = [int(num) for num in rdt.split("_")[0][1:]]
    is_skew = (rdt_as_list[2] + rdt_as_list[3]) % 2 == 1
    order = sum(rdt_as_list)
    return "skew" if is_skew else "normal", "octupole" if order == 4 else "sextupole"


def get_rdt_paths(rdts: list[str], output_dir: Path) -> dict[str, Path]:
    """Return a dictionary of RDTs and their corresponding file paths."""
    # Is there a method for this in the OMC3 source code? (jgray 2024)
    rdt_paths = {}
    for rdt in rdts:
        is_skew, order_name = get_rdt_type(rdt)
        rdt_paths[rdt] = output_dir / f"{RDT_FOLDER}/{is_skew}_{order_name}/{rdt}.tfs"
    return rdt_paths


def get_rdts_from_optics_analysis(
    beam: int,
    tbt_path: Path,
    output_dir: Path = None,
) -> dict[str, tfs.TfsDataFrame]:
    """
    Run the optics analysis for the given test parameters and return the RDTs.

    If output_dir is None, the output directory will be created in the rdt_constants.ANALYSIS_DIR.
    """
    rdts = get_rdt_names()
    only_coupling = all(rdt.lower() in ["f1001", "f1010"] for rdt in rdts)
    rdt_order = 3 # Change this to 4 if you want to include octupolar resonances
    output_dir = get_output_dir(tbt_path.name, output_dir)

    rdt_paths = get_rdt_paths(rdts, output_dir)

    # Run the analysis if the output files do not exist
    hole_in_one_entrypoint(
        files=[FREQ_OUT_DIR / tbt_path.name],
        outputdir=output_dir,
        optics=True,
        accel="lhc",
        beam=beam,
        year="2024",
        energy=6.8,
        model_dir=get_model_dir(beam),
        only_coupling=only_coupling,
        compensation="none",
        nonlinear=["rdt"],
        rdt_magnet_order=rdt_order,
    )
    tunes = get_tunes(output_dir)
    print(f"Tunes for beam {beam}: {tunes}")
    # if abs(tunes[0] - 0.28) > 0.0001 or abs(tunes[1] - 0.31) > 0.0001:
    #     raise ValueError(
    #         "Tunes are far from the expected values, rdts will be wrong/outside the tolerance"
    #     )
    dfs = {
        rdt: filter_out_BPM_near_IPs(tfs.read(path, index="NAME")) for rdt, path in rdt_paths.items()
    }
    return dfs