"""Preprocess one original top-view video without importing DeepLabCut.

Use --help for arguments. Each run requires a new output directory, so the new
geometry cannot silently overwrite movies used by existing whisker traces.
"""
from DeepWhiskerCuts.lib.top_view_preprocessing_v2 import main


if __name__ == "__main__":
    main()
