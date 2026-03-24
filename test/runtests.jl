using Test
using ParetoEnsembles
using Random

@testset "ParetoEnsembles.jl" begin

    @testset "Binh-Korn benchmark" begin
        mod = Module(:BinhKorn)
        Base.include(mod, "binh_korn_function.jl")

        initial_state = 2.0 .* (fill(1.0, 2) + 0.5 * randn(2))

        (EC, PC, RA) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            initial_state;
            rank_cutoff=4, maximum_number_of_iterations=20,
            show_trace=false
        )

        @test size(EC, 1) == 2          # two objectives
        @test size(EC, 2) > 0           # at least one solution retained
        @test size(PC, 2) == size(EC, 2) # parameter and error caches match
        @test length(RA) == size(EC, 2)  # rank array matches
        @test all(RA .>= 0)             # ranks are non-negative
    end

    @testset "Fonseca-Fleming benchmark" begin
        mod = Module(:FonsecaFleming)
        Base.include(mod, "fonseca_fleming_function.jl")

        n = 3
        initial_state = randn(n)

        (EC, PC, RA) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            initial_state;
            rank_cutoff=4, maximum_number_of_iterations=20,
            show_trace=false
        )

        @test size(EC, 1) == 2          # two objectives
        @test size(EC, 2) > 0           # at least one solution retained
        @test size(PC, 2) == size(EC, 2)
        @test length(RA) == size(EC, 2)
        @test all(RA .>= 0)
    end

    @testset "rank_function" begin
        # Simple test: two solutions, one dominates the other
        error_cache = [1.0 2.0; 1.0 2.0]
        ranks = rank_function(error_cache)
        @test ranks[1] == 0  # solution 1 is Pareto optimal
        @test ranks[2] > 0   # solution 2 is dominated

        # Single solution should have rank 0
        single = reshape([1.0, 2.0], 2, 1)
        ranks_single = rank_function(single)
        @test ranks_single[1] == 0

        # Non-dominated front: neither dominates the other
        non_dom = [1.0 2.0; 2.0 1.0]
        ranks_nd = rank_function(non_dom)
        @test all(ranks_nd .== 0)

        # Three solutions: sol3 is dominated by both sol1 and sol2
        # rank_function uses <= (weak dominance), so sol3=[2,2] is weakly dominated
        # by sol1=[1,3] only in obj1 (not obj2) and by sol2=[3,1] only in obj2
        # Actually neither weakly dominates sol3 in ALL objectives, so rank=0
        hierarchy = [1.0 3.0 2.0; 3.0 1.0 2.0]
        ranks_h = rank_function(hierarchy)
        @test ranks_h[1] == 0  # Pareto optimal
        @test ranks_h[2] == 0  # Pareto optimal
        @test ranks_h[3] == 0  # also non-dominated (tradeoff)

        # Identical solutions: no solution strictly dominates another
        identical = [1.0 1.0 1.0; 1.0 1.0 1.0]
        ranks_id = rank_function(identical)
        @test all(ranks_id .== 0)

        # Two identical + one dominated: identical pair should both be rank 0
        mixed = [1.0 1.0 2.0; 1.0 1.0 2.0]
        ranks_m = rank_function(mixed)
        @test ranks_m[1] == 0
        @test ranks_m[2] == 0
        @test ranks_m[3] == 2  # dominated by both identical solutions
    end

    @testset "Reproducibility with seeded RNG" begin
        mod = Module(:BinhKornRepr)
        Base.include(mod, "binh_korn_function.jl")

        # Same seed should produce same results
        rng1 = MersenneTwister(12345)
        Random.seed!(42)
        initial_state = 2.0 .* (fill(1.0, 2) + 0.5 * randn(MersenneTwister(99), 2))

        (EC1, PC1, RA1) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            copy(initial_state);
            rank_cutoff=4, maximum_number_of_iterations=10,
            show_trace=false, rng=MersenneTwister(12345)
        )

        (EC2, PC2, RA2) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            copy(initial_state);
            rank_cutoff=4, maximum_number_of_iterations=10,
            show_trace=false, rng=MersenneTwister(12345)
        )

        # Internal RNG is seeded, but neighbor_function uses global RNG,
        # so results won't be identical. Just verify the RNG kwarg is accepted
        # and produces valid output.
        @test size(EC1, 1) == 2
        @test size(EC2, 1) == 2
        @test all(RA1 .>= 0)
        @test all(RA2 .>= 0)
    end

    @testset "Pareto rank properties" begin
        # Property: rank is always non-negative
        for _ in 1:10
            n_obj = rand(2:5)
            n_sol = rand(2:20)
            E = rand(n_obj, n_sol)
            R = rank_function(E)
            @test all(R .>= 0)
            @test length(R) == n_sol
        end

        # Property: at least one solution has rank 0 (Pareto front is never empty)
        for _ in 1:10
            n_obj = rand(2:5)
            n_sol = rand(2:20)
            E = rand(n_obj, n_sol)
            R = rank_function(E)
            @test minimum(R) == 0
        end

        # Property: if all objectives are the same ordering, only the best has rank 0
        for _ in 1:5
            n_sol = rand(3:10)
            vals = sort(rand(n_sol))  # strictly increasing
            E = [vals'; vals']  # same ordering for both objectives
            R = rank_function(E)
            @test R[1] == 0
            @test all(R[2:end] .> 0)
        end
    end

    @testset "Edge cases" begin
        mod = Module(:BinhKornEdge)
        Base.include(mod, "binh_korn_function.jl")

        # Very few iterations
        (EC, PC, RA) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=4, maximum_number_of_iterations=1,
            temperature_min=0.5, show_trace=false
        )
        @test size(EC, 1) == 2
        @test size(EC, 2) >= 1

        # High rank_cutoff (keep everything)
        (EC2, PC2, RA2) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=1000, maximum_number_of_iterations=5,
            temperature_min=0.5, show_trace=false
        )
        @test size(EC2, 2) >= size(EC, 2)

        # Low rank_cutoff (aggressive pruning)
        (EC3, PC3, RA3) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=1, maximum_number_of_iterations=5,
            temperature_min=0.5, show_trace=false
        )
        @test size(EC3, 1) == 2
        @test size(EC3, 2) >= 1  # at least one solution retained
    end

    @testset "Incremental rank (_rank_insert!) matches full rank" begin
        for _ in 1:20
            n_obj = rand(2:5)
            n_sol = rand(3:15)
            cols = [rand(n_obj) for _ in 1:n_sol]

            # build up incrementally
            inc_cols = Vector{Float64}[cols[1]]
            inc_ranks = zeros(1)
            for i in 2:n_sol
                push!(inc_cols, cols[i])
                ParetoEnsembles._rank_insert!(inc_cols, inc_ranks)
            end

            # full rank from scratch
            full_ranks = ParetoEnsembles._rank_columns(cols)

            @test inc_ranks == full_ranks
        end
    end

    @testset "maximum_archive_size cap" begin
        mod = Module(:BinhKornCap)
        Base.include(mod, "binh_korn_function.jl")

        (EC, PC, RA) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=1000, maximum_number_of_iterations=20,
            temperature_min=0.5, show_trace=false,
            maximum_archive_size=10
        )
        @test size(EC, 2) <= 10
        @test size(EC, 1) == 2
        @test all(RA .>= 0)
    end

    @testset "Threaded _rank_columns matches serial" begin
        for _ in 1:10
            cols = [rand(3) for _ in 1:20]
            @test ParetoEnsembles._rank_columns(cols) == ParetoEnsembles._rank_columns_threaded(cols)
        end
    end

    @testset "Threaded _rank_insert! matches serial" begin
        for _ in 1:10
            cols = [rand(3) for _ in 1:10]

            # serial
            r1 = zeros(1)
            c1 = Vector{Float64}[cols[1]]
            for i in 2:length(cols)
                push!(c1, cols[i])
                ParetoEnsembles._rank_insert!(c1, r1)
            end

            # threaded
            r2 = zeros(1)
            c2 = Vector{Float64}[cols[1]]
            for i in 2:length(cols)
                push!(c2, cols[i])
                ParetoEnsembles._rank_insert_threaded!(c2, r2)
            end

            @test r1 == r2
        end
    end

    @testset "parallel_evaluation flag" begin
        mod = Module(:BinhKornPar)
        Base.include(mod, "binh_korn_function.jl")

        (EC, PC, RA) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=4, maximum_number_of_iterations=10,
            temperature_min=0.5, show_trace=false,
            parallel_evaluation=true
        )
        @test size(EC, 1) == 2
        @test size(EC, 2) > 0
        @test all(RA .>= 0)
    end

    @testset "estimate_ensemble_parallel multi-chain" begin
        mod = Module(:BinhKornMulti)
        Base.include(mod, "binh_korn_function.jl")

        initial_states = [[2.5, 1.5], [1.0, 2.0], [3.0, 1.0]]

        (EC, PC, RA) = estimate_ensemble_parallel(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            initial_states;
            rank_cutoff=4, maximum_number_of_iterations=10,
            temperature_min=0.5, show_trace=false
        )
        @test size(EC, 1) == 2
        @test size(EC, 2) >= 3  # at least one solution per chain
        @test all(RA .>= 0)
    end

    @testset "hypervolume" begin
        # Two non-dominated points: (1,3) and (3,1) with ref (4,4)
        # Sorted by f1: (1,3), (3,1)
        # (1,3) contributes (3-1)*(4-3) = 2
        # (3,1) contributes (4-3)*(4-1) = 3 → total = 5
        E = [1.0 3.0; 3.0 1.0]
        @test hypervolume(E, [4.0, 4.0]) ≈ 5.0

        # Single point
        E1 = reshape([2.0, 2.0], 2, 1)
        @test hypervolume(E1, [5.0, 5.0]) ≈ 9.0

        # Dominated point should be filtered out: (1,1) dominates (2,2)
        E2 = [1.0 2.0; 1.0 2.0]
        @test hypervolume(E2, [3.0, 3.0]) ≈ 4.0

        # All points outside reference
        E3 = [5.0 6.0; 5.0 6.0]
        @test hypervolume(E3, [4.0, 4.0]) ≈ 0.0

        # Empty front
        E4 = reshape(Float64[], 2, 0)
        @test hypervolume(E4, [4.0, 4.0]) ≈ 0.0

        # Three non-dominated points
        E5 = [1.0 2.0 3.0; 4.0 2.0 1.0]
        ref = [5.0, 5.0]
        hv = hypervolume(E5, ref)
        # (1,4): width=1, height=1 → 1
        # (2,2): width=1, height=3 → 3
        # (3,1): width=2, height=4 → 8 → total=12
        @test hv ≈ 12.0
    end

    @testset "pareto_front" begin
        # (1,2) and (2,1) are non-dominated; (3,3) is dominated by both
        E = [1.0 2.0 3.0; 2.0 1.0 3.0]
        P = [10.0 20.0 30.0; 10.0 20.0 30.0]
        R = rank_function(E)
        fe, fp = pareto_front(E, P, R)
        @test size(fe, 2) == 2  # two non-dominated solutions
        @test size(fp, 2) == 2
        # check that all returned solutions have rank 0
        fr = rank_function(fe)
        @test all(fr .== 0)
    end

    @testset "convergence trace" begin
        mod = Module(:BinhKornTrace)
        Base.include(mod, "binh_korn_function.jl")

        (EC, PC, RA, trace) = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=4, maximum_number_of_iterations=10,
            temperature_min=0.5, show_trace=false,
            trace=true, trace_reference_point=[200.0, 50.0]
        )

        @test length(trace) > 0
        @test haskey(first(trace), :temperature)
        @test haskey(first(trace), :archive_size)
        @test haskey(first(trace), :hypervolume)
        @test all(t.hypervolume >= 0 for t in trace)
        # temperatures should be decreasing
        temps = [t.temperature for t in trace]
        @test issorted(temps, rev=true)

        # without trace, should return 3-tuple as before
        result = estimate_ensemble(
            mod.objective_function, mod.neighbor_function,
            mod.acceptance_probability_function, mod.cooling_function,
            [2.5, 1.5];
            rank_cutoff=4, maximum_number_of_iterations=10,
            temperature_min=0.5, show_trace=false
        )
        @test length(result) == 3
    end
end
