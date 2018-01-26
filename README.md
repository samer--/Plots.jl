This is a fork of https://github.com/JuliaPlots/Plots.jl
Differences from upstream include:


Fixes
- Pick up defaults from `PLOTS_DEFAULTS` when changing theme.
- Allow backend to report its physical pixel dimension BEFORE doing layout,
  so that the `px` unit can be set-up correctly.
- GR backend now uses correct work
- Now using `with_tempname()` to clean up temporary files properly.

Enhancements
- Added beginnings of a framework for multiple figures, currently only support by GR backend.
  This is assisted by a refactoring of `output.jl` and a change to the interface to the
  backends.
- GR backend layout is made more consistent and reliable by measuring all pieces of text
  to compute margins and placement.
- GR backend supports multiple figures by opening and keeping track of multiple GR workstations.
- Ticks in GR backend now have an absolute length, not proportional to dimensions of plot.
- GR backend fills whole window with background colour, not just the WS viewport.
- Several rendering attributes in GR backend (eg frame line width, axis tick length, gap between 
  axis tick labels) are now named constants, in advance of surfacing them as user specified
  attributes.
- Marker rendering in GR rewritten and optimised to suit marker rendering in my fork of GR.
  It now tries to satisfy marker alpha and marker stroke width and alpha requirements
  by choosing from 3 marker drawing methods.
- In GR, marker sizes and linewidths are interpreted as physical dimensions in points.

Refactorings
- Axes layout and rendering in GR
- GR no longer relies on manipulating environment ENV to control GR library.
- `_before_layout_calcs` removed from interface to backends, now backends do what they
  like in `_display` and `_show` but should call back to `prepare_output` to layout subplots.

NB. Many of these features rely on my fork of GR, at https://github.com/samer--/gr

# Plots

[![Build Status](https://travis-ci.org/JuliaPlots/Plots.jl.svg?branch=master)](https://travis-ci.org/JuliaPlots/Plots.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/github/juliaplots/plots.jl?branch=master&svg=true)](https://ci.appveyor.com/project/mkborregaard/plots-jl)
[![Join the chat at https://gitter.im/tbreloff/Plots.jl](https://badges.gitter.im/tbreloff/Plots.jl.svg)](https://gitter.im/tbreloff/Plots.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
<!-- [![Plots](http://pkg.julialang.org/badges/Plots_0.3.svg)](http://pkg.julialang.org/?pkg=Plots&ver=0.3) -->
<!-- [![Plots](http://pkg.julialang.org/badges/Plots_0.4.svg)](http://pkg.julialang.org/?pkg=Plots&ver=0.4) -->
<!-- [![Coverage Status](https://coveralls.io/repos/tbreloff/Plots.jl/badge.svg?branch=master)](https://coveralls.io/r/tbreloff/Plots.jl?branch=master) -->
<!-- [![codecov.io](http://codecov.io/github/tbreloff/Plots.jl/coverage.svg?branch=master)](http://codecov.io/github/tbreloff/Plots.jl?branch=master) -->

#### Created by Tom Breloff (@tbreloff)

#### Maintained by the [JuliaPlot members](https://github.com/orgs/JuliaPlots/people)

Plots is a plotting API and toolset.  My goals with the package are:

- **Powerful**.  Do more with less.  Complex visualizations become easy.
- **Intuitive**.  Stop reading so much documentation.  Commands should "just work".
- **Concise**.  Less code means fewer mistakes and more efficient development/analysis.
- **Flexible**.  Produce your favorite plots from your favorite package, but quicker and simpler.
- **Consistent**.  Don't commit to one graphics package, use the same code everywhere.
- **Lightweight**.  Very few dependencies.
- **Smart**. Attempts to figure out what you **want** it to do... not just what you **tell** it.

View the [full documentation](http://docs.juliaplots.org/latest).
