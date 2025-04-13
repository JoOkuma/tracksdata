from pathlib import Path

import numpy as np

from tifffile import imread

from tracksdata.graph._rustworkx_graph import RustWorkXGraphBackend
from tracksdata.nodes._regionprops import RegionPropsOperator


def main() -> None:

    # load from HeLa
    data_dir = Path("examples/Fluo-N2DL-HeLa/01_GT/TRA")
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."

    labels = np.stack(
        [
            imread(p)
            for p in sorted(data_dir.glob("*.tif"))
        ],
    )

    graph = RustWorkXGraphBackend()
    nodes_operator = RegionPropsOperator(graph, show_progress=True)

    nodes_operator(labels=labels)
    

if __name__ == "__main__":
    main()
