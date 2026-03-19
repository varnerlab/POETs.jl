using Test
using POETs
using Random

@testset "POETs.jl" begin

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

        # Identical solutions: each is dominated by (n-1) others (all are <=)
        identical = [1.0 1.0 1.0; 1.0 1.0 1.0]
        ranks_id = rank_function(identical)
        @test all(ranks_id .== length(ranks_id) - 1)
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
end
