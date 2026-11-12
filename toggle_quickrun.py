import nbformat as nbf
for name in ['UDL_Group91_Assignment2_PartA.ipynb', 'UDL_Group91_Assignment2_PartB.ipynb']:
    nb = nbf.read(name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'QUICK_RUN = False' in cell.source:
            cell.source = cell.source.replace('QUICK_RUN = False', 'QUICK_RUN = True')
    nbf.write(nb, name.replace('.ipynb', '_quicksrc.ipynb'))
