"""
ui -- the windows (tkinter + matplotlib), and the plumbing only they need.

``tkbase``         forms, progress / worker thread, the image panel, colormaps
``data``           the dataset objects the viewers work on (no GUI)
``viewers``        cut, map contour + its cuts, spatial scan, pop-out windows
``analysis``       k conversion, arbitrary cut, FS correction, Fermi level,
                   data operations, Brillouin zone
``curves``         the curve viewer, curve fit, spin analysis
``fit``            MDC / EDC fitting, dispersion, self-energy
``process``        2-D processing and the stack plot
``volume``         3-D processing and the 3-D view
``figure``         the figure composer
``cutops``         cut arithmetic
``degrid``         removing the detector grid
``kzmap``          kz map processing (per-spectrum Fermi level)
``kzconv``         kz map -> momentum
``loader_dialog``  the Load-data window
``list_actions``   which right-click entries fit a selection (no GUI)

This is the only package that imports tkinter / the matplotlib Tk backend.
"""
