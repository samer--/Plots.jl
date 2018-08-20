bind(f,a1) = (args...) -> f(a1, args...)
fork(f,g) = x -> (f(x), g(x))
id(x) = x

function with_tempname(action, ext="")
    f = tempname() * ext
    try     return action(f)
    finally rm(f, force=true) end
end

# ------ saving in various formats, using backend interface ---
plot_writer(mime, plt) = io -> _show(io, MIME(mime), plt)
save_as(mime, plt, fn) = open(plot_writer(mime, plt), fn, "w")

# these are exported by Plots.
png(plt::Plot, fn::AbstractString) = save_as("image/png", plt, addExtension(fn, "png"))
png(fn::AbstractString) = png(current(), fn)

# ---- fallback implementations of _show ----
# if no PNG generation is defined, make PDF and use FileIO to convert
if is_installed("FileIO")
    @eval import FileIO
    function _show(io::IO, ::MIME"image/png", plt::Plot)
        with_tempname() do pdf_file
            save_as("application/pdf", plt, pdf_file)
            with_tempname() do png_file
                FileIO.save(png_file, FileIO.load(pdf_file))
                write(io, readstring(open(png_file)))
            end
        end
    end
end

# ----------------------------------------------------------------

function html(ext::String, plt::Plot, fn::AbstractString)
    _use_remote[] = true
    save_as("text/html", plt, fn)
    _use_remote[] = false
end

const _savemap = Dict(
    "png"  => bind(save_as, "image/png"),
    "svg"  => bind(save_as, "image/svg+xml"),
    "pdf"  => bind(save_as, "application/pdf"),
    "ps"   => bind(save_as, "application/postscript"),
    "eps"  => bind(save_as, "image/eps"),
    "tex"  => bind(save_as, "application/x-tex"),
    "html" => html
  )

function getExtension(fn::AbstractString)
  pieces = split(fn, ".")
  length(pieces) > 1 || error("Can't extract file extension: ", fn)
  ext = pieces[end]
  haskey(_savemap, ext) || error("Invalid file extension: ", fn)
  ext
end

function addExtension(fn::AbstractString, ext::AbstractString)
  try getExtension(fn) == ext ? fn : "$fn.$ext" catch "$fn.$ext" end
end

defaultOutputFormat(plt::Plot) = "png"

"""
    savefig([plot,] filename)

Save a Plot (the current plot if `plot` is not passed) to file. The file
type is inferred from the file extension. All backends support png and pdf
file types, some also support svg, ps, eps, html and tex.
"""
function savefig(plt::Plot, fn::AbstractString)
  (fn1, ext) = try fork(id, getExtension)(fn)
               catch fork(bind(addExtension, fn), id)(defaultOutputFormat(plt))
               end
  # save it
  func = get(_savemap, ext) do
    error("Unsupported extension $ext with filename ", fn1)
  end
  func(plt, fn1)
end
savefig(fn::AbstractString) = savefig(current(), fn)

"""
    gui([plot])

Display a plot using the backends' default gui window
"""
gui(plt::Plot) = _display(plt)

"""
    gui(fig, plot)

Display a plot using the specified backends' gui window, if supported.
"""
gui(fig::Any, plt::Plot) = _display(fig, plt)

"Close all open gui windows of the current backend"
closeall() = closeall(backend())

# ---------------------------------------------------------
# Hooks into various display mechanisms

# override the REPL display to open a gui window
Base.display(::Base.REPL.REPLDisplay, ::MIME"text/plain", plt::Plot) = gui(plt)
Base.display(::PlotsDisplay, plt::Plot) = gui(plt)

# ---------------------------------------------------------

const _mimeformats = [
    "application/eps",
    "image/eps",
    "application/pdf",
    "image/png",
    "application/postscript",
    "image/svg+xml",
    "text/plain",
    "text/html",
    "application/x-tex"]

const _best_html_output_type = KW(
    :pyplot => :png,
    :unicodeplots => :txt,
    :glvisualize => :png,
    :plotlyjs => :html,
    :plotly => :html
)

# a backup for html... passes to svg or png depending on the html_output_format arg
function _show(io::IO, ::MIME"text/html", plt::Plot)
    output_type = Symbol(plt.attr[:html_output_format])
    if output_type == :auto
        output_type = get(_best_html_output_type, backend_name(plt.backend), :svg)
    end
    if output_type == :png
        print(io, "<img src=\"data:image/png;base64,", base64encode(plot_writer("image/png", plt)), "\" />")
    elseif output_type == :svg
        show(io, MIME("image/svg+xml"), plt)
    elseif output_type == :txt
        show(io, MIME("text/plain"), plt)
    else
        error("only png or svg allowed. got: $output_type")
    end
end

# for writing to io streams... first prepare, then callback
for mime in _mimeformats
    @eval Base.show(io::IO, m::MIME{Symbol($mime)}, plt::Plot{B}) where B  = _show(io, m, plt)
end

# IJulia only... inline display
function inline(plt::Plot = current())
    if !isijulia() error("inline() is IJulia-only") end
    Main.IJulia.clear_output(true)
    display(Main.IJulia.InlineDisplay(), plt)
end


# ---------------------------------------------------------
# IJulia
# ---------------------------------------------------------

@require IJulia begin
    if IJulia.inited

        function IJulia.display_dict(plt::Plot)
            best_type() = get(_best_html_output_type, backend_name(plt.backend), :svg) 
            requested_type = Symbol(plt.attr[:html_output_format])
            output_type    = if requested_type == :auto best_type() else requested_type end

            ok(mime, fn) = () -> Dict{String,String}(mime => fn(plot_writer(mime, plt)))
            unsupported  = () -> error("Unsupported output type $output_type")
            get(Dict(:png  => ok("image/png", base64encode),
                     :svg  => ok("image/svg+xml", sprint),
                     :html => ok("text/html", sprint)), 
                output_type, unsupported)()
        end

        # default text/plain passes to html... handles Interact issues
        Base.show(io::IO, m::MIME"text/plain", plt::Plot) = _show(io, MIME"text/html", plt)

        ENV["MPLBACKEND"] = "Agg"
    end
end

# ---------------------------------------------------------
# Atom PlotPane
# ---------------------------------------------------------
@require Juno begin
    import Hiccup, Media

    if Juno.isactive()
        Media.media(Plot, Media.Plot)

        _show(io::IO, m::MIME"text/plain", plt::Plot{B}) where {B} = print(io, "Plot{$B}()")

        function Juno.render(e::Juno.Editor, plt::Plot)
            Juno.render(e, nothing)
        end

        if get(ENV, "PLOTS_USE_ATOM_PLOTPANE", true) in (true, 1, "1", "true", "yes")
            function Juno.render(pane::Juno.PlotPane, plt::Plot)
                # temporarily overwrite size to be Atom.plotsize
                sz = plt[:size]
                jsize = Juno.plotsize()
                if jsize[1] == 0 jsize[1] = 400 end
                if jsize[2] == 0 jsize[2] = 500 end
                plt[:size] = jsize
                Juno.render(pane, HTML(stringmime(MIME("text/html"), plt)))
                plt[:size] = sz
            end
            # special handling for PlotlyJS
            function Juno.render(pane::Juno.PlotPane, plt::Plot{PlotlyJSBackend})
                display(Plots.PlotsDisplay(), plt)
            end
        else
            function Juno.render(pane::Juno.PlotPane, plt::Plot)
                display(Plots.PlotsDisplay(), plt)
                s = "PlotPane turned off.  Unset ENV[\"PLOTS_USE_ATOM_PLOTPANE\"] and restart Julia to enable it."
                Juno.render(pane, HTML(s))
            end
        end

        # special handling for plotly... use PlotsDisplay
        function Juno.render(pane::Juno.PlotPane, plt::Plot{PlotlyBackend})
            display(Plots.PlotsDisplay(), plt)
            s = "PlotPane turned off.  The plotly backend cannot render in the PlotPane due to javascript issues. Plotlyjs is similar to plotly and is compatible with the plot pane."
            Juno.render(pane, HTML(s))
        end
    end
end
