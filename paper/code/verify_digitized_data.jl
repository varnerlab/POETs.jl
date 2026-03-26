# Verify digitized Butenas 1999 data against Hockin-Mann model predictions
#
# The Hockin-Mann model was calibrated to reproduce the Butenas 1999
# synthetic plasma experiments. So model predictions at nominal parameters
# should closely match the experimental data. Large discrepancies flag
# digitization errors.
#
# Run from the paper/code directory:
#   julia --project verify_digitized_data.jl

using HockinMannModel
using Statistics

# ──────────────────────────────────────────────────────────────
# Load digitized data
# ──────────────────────────────────────────────────────────────
datafile = joinpath(@__DIR__, "data", "butenas1999_prothrombin.csv")
lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(datafile))
data_rows = [split(l, ",") for l in lines[2:end]]

conditions = ["FII_50pct", "FII_75pct", "FII_100pct", "FII_125pct", "FII_150pct"]
fractions = [0.50, 0.75, 1.00, 1.25, 1.50]
labels = ["50%", "75%", "100%", "125%", "150%"]

exp_data = Dict{String, NamedTuple{(:t, :thr), Tuple{Vector{Float64}, Vector{Float64}}}}()
for cond in conditions
    rows = filter(r -> strip(r[3]) == cond, data_rows)
    t = [parse(Float64, r[1]) for r in rows]
    thr = [parse(Float64, r[2]) for r in rows]
    exp_data[cond] = (t=t, thr=thr)
end

# ──────────────────────────────────────────────────────────────
# Simulate Hockin-Mann at each prothrombin level
# ──────────────────────────────────────────────────────────────
P_NOM = default_rate_constants(HockinMann2002)

println("Verifying digitized data against Hockin-Mann model predictions")
println("="^75)
println()

for (cond, frac, label) in zip(conditions, fractions, labels)
    d = exp_data[cond]

    # Simulate at this prothrombin level
    u0 = default_initial_conditions(HockinMann2002; TF_concentration=5e-12)
    u0[14] = 1.4e-6 * frac  # scale prothrombin

    sol = simulate(HockinMann2002;
        TF_concentration = 5e-12,
        tspan = (0.0, 14.0 * 60.0),
        saveat = 60.0,  # every minute
        p = P_NOM, u0 = u0)

    t_model_min = sol.t ./ 60.0
    thr_model_nM = total_thrombin(HockinMann2002, sol) .* 1e9

    # Compare at matching time points
    println("$label prothrombin (FII = $(frac*1400) nM):")
    println("  Time(min)  Digitized(nM)  Model(nM)    Diff(nM)   Diff(%)")
    println("  " * "-"^65)

    peak_dig = maximum(d.thr)
    peak_model = maximum(thr_model_nM)
    tpeak_dig = d.t[argmax(d.thr)]
    tpeak_model = t_model_min[argmax(thr_model_nM)]

    for (t_exp, thr_exp) in zip(d.t, d.thr)
        # Find closest model time point
        idx = argmin(abs.(t_model_min .- t_exp))
        thr_mod = thr_model_nM[idx]
        diff = thr_exp - thr_mod
        pct = thr_mod > 1.0 ? round(diff / thr_mod * 100, digits=1) : "-"
        flag = (abs(diff) > 50 && thr_mod > 10) ? " <<<" : ""
        println("  $(lpad(t_exp, 6))     $(lpad(round(thr_exp, digits=1), 10))  $(lpad(round(thr_mod, digits=1), 10))  $(lpad(round(diff, digits=1), 10))  $(rpad(pct, 8))$flag")
    end

    println()
    println("  PEAK: digitized=$(round(peak_dig, digits=1)) nM at $(tpeak_dig) min")
    println("        model=$(round(peak_model, digits=1)) nM at $(tpeak_model) min")
    diff_pct = round(abs(peak_dig - peak_model) / peak_model * 100, digits=1)
    println("        difference: $(diff_pct)%")
    println()
end

# Summary: Table 2 cross-check
println("="^75)
println("TABLE 2 CROSS-CHECK (Max IIa as % of 100% reference)")
println("="^75)
ref_peak_dig = maximum(exp_data["FII_100pct"].thr)
ref_peak_model = let
    u0 = default_initial_conditions(HockinMann2002; TF_concentration=5e-12)
    sol = simulate(HockinMann2002;
        TF_concentration = 5e-12, tspan = (0.0, 14.0*60.0),
        saveat = 60.0, p = P_NOM, u0 = u0)
    maximum(total_thrombin(HockinMann2002, sol)) * 1e9
end

println()
println("  Condition  | Digitized peak | Model peak | Table 2 ratio | Dig ratio | Model ratio")
println("  " * "-"^80)
for (cond, frac, label) in zip(conditions, fractions, labels)
    peak_dig = maximum(exp_data[cond].thr)
    u0 = default_initial_conditions(HockinMann2002; TF_concentration=5e-12)
    u0[14] = 1.4e-6 * frac
    sol = simulate(HockinMann2002;
        TF_concentration = 5e-12, tspan = (0.0, 14.0*60.0),
        saveat = 60.0, p = P_NOM, u0 = u0)
    peak_model = maximum(total_thrombin(HockinMann2002, sol)) * 1e9

    dig_ratio = round(peak_dig / ref_peak_dig * 100, digits=0)
    mod_ratio = round(peak_model / ref_peak_model * 100, digits=0)

    # Table 2 values (from Butenas 1999, no protein C, PCPS)
    table2 = if frac == 0.5; "50%" elseif frac == 1.5; "195%" else "—" end

    println("  $(rpad(label, 11)) | $(lpad(round(peak_dig, digits=0), 12)) nM | $(lpad(round(peak_model, digits=0), 8)) nM | $(lpad(table2, 12))  | $(lpad(dig_ratio, 7))%  | $(lpad(mod_ratio, 9))%")
end

println()
println("If 'Dig ratio' matches 'Table 2 ratio', digitization peaks are consistent.")
println("If 'Model ratio' matches 'Table 2 ratio', model reproduces the experimental scaling.")
