# Shared publication-quality theme for all paper figures
# Include this file before generating any figure.

using CairoMakie

# ──────────────────────────────────────────────────────────────
# Color palette
# ──────────────────────────────────────────────────────────────
const C_FRONT = RGBAf(0.05, 0.05, 0.05, 0.9)           # near-black for Pareto front
const C_ENSEMBLE = RGBAf(0.20, 0.45, 0.78, 0.55)        # steel blue for ensemble cloud
const C_ENSEMBLE_FILL = RGBAf(0.20, 0.45, 0.78, 0.15)   # light fill for CI bands
const C_MEAN = RGBf(0.20, 0.45, 0.78)                   # solid blue for ensemble mean
const C_DATA = RGBf(0.1, 0.1, 0.1)                      # black for data points
const C_TRUE = RGBf(0.1, 0.1, 0.1)                      # black for true trajectories
const C_THEORY = RGBf(0.85, 0.15, 0.15)                 # red for theoretical fronts
const C_NSGA = RGBf(0.85, 0.35, 0.10)                   # orange for NSGA-II
const C_PE = RGBf(0.20, 0.45, 0.78)                     # blue for ParetoEnsembles

# Hemophilia colors
const C_NORMAL = RGBf(0.20, 0.45, 0.78)
const C_MILD = RGBf(0.90, 0.55, 0.10)
const C_SEVERE = RGBf(0.80, 0.15, 0.15)

# ──────────────────────────────────────────────────────────────
# Publication theme
# ──────────────────────────────────────────────────────────────
function paper_theme()
    Theme(
        fontsize = 18,
        font = "CMU Serif",
        Axis = (
            backgroundcolor = RGBf(0.96, 0.96, 0.96),
            xgridvisible = true,
            ygridvisible = true,
            xgridcolor = RGBAf(0.5, 0.5, 0.5, 0.3),
            ygridcolor = RGBAf(0.5, 0.5, 0.5, 0.3),
            xgridwidth = 1.0,
            ygridwidth = 1.0,
            xminorgridvisible = false,
            yminorgridvisible = false,
            spinewidth = 0.8,
            xtickwidth = 0.8,
            ytickwidth = 0.8,
            xlabelsize = 20,
            ylabelsize = 20,
            titlesize = 20,
            xticklabelsize = 17,
            yticklabelsize = 17,
        ),
        Legend = (
            framevisible = false,
            labelsize = 17,
            patchsize = (20, 14),
        ),
    )
end

function set_paper_theme!()
    set_theme!(paper_theme())
end

# ──────────────────────────────────────────────────────────────
# Helper: rank-based coloring
# ──────────────────────────────────────────────────────────────
function rank_colors(RA)
    max_rank = max(maximum(RA), 1)
    colors = Vector{Any}(undef, length(RA))
    for i in eachindex(RA)
        if RA[i] == 0
            colors[i] = C_FRONT
        else
            t = clamp((RA[i] - 1) / max(max_rank - 1, 1), 0, 1)
            colors[i] = RGBAf(0.20 + 0.40*t, 0.45 + 0.25*t, 0.78 - 0.20*t, 0.55 - 0.25*t)
        end
    end
    return colors
end
