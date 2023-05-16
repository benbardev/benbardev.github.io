Example of converting jupyter notebook to markdown with removed cells.

jupyter nbconvert --to=markdown cloud_learner.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove-cell
