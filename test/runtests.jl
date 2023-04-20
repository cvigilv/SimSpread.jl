using Test
using SimSpread
using NamedArrays
using MLBase

@testset "General utilities" verbose = true begin
    @testset "read_namedmatrix" begin
        W = NamedArray(zeros(2, 3), (["s1", "s2"], ["t1", "t2", "t3"]))
        X = NamedArray(zeros(2, 3), (["s1", "s2"], ["C#1", "C#2", "C#3"]))
        Y = NamedArray(zeros(2, 3), (["R#1", "R#2"], ["t1", "t2", "t3"]))
        Z = NamedArray(zeros(2, 3), (["R#1", "R#2"], ["C#1", "C#2", "C#3"]))

        @test SimSpread.read_namedmatrix("data/mat1") == X
        @test SimSpread.read_namedmatrix("data/mat2"; cols=false) == Y
        @test SimSpread.read_namedmatrix("data/mat3"; rows=false) == Z
        @test SimSpread.read_namedmatrix("data/mat4"; rows=false, cols=false) == Z
    end

    @testset "k" begin
        M = [0 0 0; 0 0 1; 0 1 1; 1 1 1]

        @test k(1, M) ≈ 0
        @test k(M[1, :]) ≈ 0
        @test k(M) ≈ [0, 1, 2, 3]
    end
end

@testset verbose = true "SimSpread Core" begin
    @testset "split" begin
        X = [["s4", "s3"], ["s9", "s10"], ["s5", "s6"], ["s1", "s2"], ["s7", "s8"]]
        X̂ = NamedArray(zeros(10, 5), (["s$i" for i in 1:10], ["t$i" for i in 1:5]))

        @test X == SimSpread.split(X̂, 5; seed=1)
    end
    @testset "cutoff" begin
        # Prepare test
        x = 0.8
        y = hcat(collect(0.0:0.1:1.0))
        z = [0.1 0.5; 0.5 1.0]

        @testset "Cutoff = mean(values)" begin
            α = 0.5
            @test cutoff(x, α, false) ≈ 1.0
            @test cutoff(x, α, true) ≈ 0.8
            @test cutoff(y, α, false) ≈ [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            @test cutoff(y, α, true) ≈ [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            @test cutoff(z, α, false) ≈ [0.0 1.0; 1.0 1.0]
            @test cutoff(z, α, true) ≈ [0.0 0.5; 0.5 1.0]
        end

        @testset "Cutoff < min(values)" begin
            β = -0.01
            @test cutoff(x, β, false) ≈ 1.0
            @test cutoff(x, β, true) ≈ x
            @test cutoff(y, β, false) ≈ ones(Float64, 11)
            @test cutoff(y, β, true) ≈ y
            @test cutoff(z, β, false) ≈ ones(Float64, 2, 2)
            @test cutoff(z, β, true) ≈ z
        end

        @testset "Cutoff > max(values)" begin
            γ = 1.01
            @test cutoff(x, γ, false) ≈ 0.0
            @test cutoff(x, γ, true) ≈ 0.0
            @test cutoff(y, γ, false) ≈ zeros(Float64, 11)
            @test cutoff(y, γ, true) ≈ zeros(Float64, 11)
            @test cutoff(z, γ, false) ≈ zeros(Float64, 2, 2)
            @test cutoff(z, γ, true) ≈ zeros(Float64, 2, 2)
        end
    end

    @testset "featurize" begin
        M₀ = NamedArray([0.1 0.5; 0.5 1.0], (["s1", "s2"], ["s1", "s2"]))
        M₁ = NamedArray([0.0 1.0; 1.0 1.0], (["s1", "s2"], ["fs1", "fs2"]))
        M₂ = NamedArray([0.0 0.5; 0.5 1.0], (["s1", "s2"], ["fs1", "fs2"]))
        α = 0.5

        @test featurize(M₀, α, false) == M₁
        @test featurize(M₀, α, true) == M₂
    end

    @testset "prepare" begin
        # Prepare test
        DF = NamedArray([1 0 1; 1 1 0; 0 1 1])
        DT = NamedArray([0 1; 1 1; 1 0])
        C = ["D1"]
        setnames!(DF, ["D$i" for i in 1:3], 1)
        setnames!(DF, ["fD$i" for i in 1:3], 2)
        setnames!(DT, ["D$i" for i in 1:3], 1)
        setnames!(DT, ["T$i" for i in 1:2], 2)

        @testset "Graph construction" begin
            A, B = prepare(DT, DF, C)
            @test all(names(A, 1) .== names(A, 2))
            @test all(names(B, 1) .== names(B, 2))
            @test all(names(A, 1) .== ["D1", "D2", "D3", "fD2", "fD3", "T1", "T2"])
            @test all(names(B, 1) .== ["D1", "D2", "D3", "fD2", "fD3", "T1", "T2"])
        end

        @testset "Graph construction errors" begin
            setnames!(DF, ["D$i" for i in 1:3], 2)
            @test_throws AssertionError("Features and drugs have the same names!") prepare(DT, DF, C)
        end
    end

    @testset "spread" begin
        M = [1.0 0.0 0.0; 1.0 1.0 0.0; 1.0 1.0 1.0]
        W = [1.0 0.0 0.0; 0.5 0.5 0.0; 0.33333 0.33333 0.33333]

        @test spread(M) ≈ W rtol=10^-5
    end

    @testset "predict" begin
        A = NamedArray(Matrix{Float64}([
                0 0 0 0 0 0 1 0 0
                0 0 0 0 1 1 0 1 0
                0 0 0 0 1 1 0 1 0
                0 0 0 0 0 0 1 0 1
                0 1 1 0 0 0 0 0 0
                0 1 1 0 0 0 0 0 0
                1 0 0 1 0 0 0 0 0
                0 1 1 0 0 0 0 0 0
                0 0 0 1 0 0 0 0 0
            ]),
            (
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
            )
        )
        B = NamedArray(Matrix{Float64}([
                0 0 0 0 0 0 0 0 0
                0 0 0 0 1 1 0 1 0
                0 0 0 0 1 1 0 1 0
                0 0 0 0 0 0 1 0 1
                0 1 1 0 0 0 0 0 0
                0 1 1 0 0 0 0 0 0
                0 0 0 1 0 0 0 0 0
                0 1 1 0 0 0 0 0 0
                0 0 0 1 0 0 0 0 0
            ]),
            (
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
            )
        )
        yhat = NamedArray(Matrix{Float64}([
                0 0 0 0 0 0 0.5 0 0.5
                0 0 0 0 1 1 0 1 0
                0 0 0 0 1 1 0 1 0
                0 0 0 0 0 0 1 0 1
                0 1 1 0 0 0 0 0 0
                0 1 1 0 0 0 0 0 0
                0 0 0 1 0 0 0 0 0
                0 1 1 0 0 0 0 0 0
                0 0 0 1 0 0 0 0 0
            ]),
            (
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
            )
        )
        y = NamedArray(Matrix{Float64}([0 1]), (["q1"], ["t1", "t2"]))

        @test SimSpread.predict(
            A,
            B,
            (
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
                ["q1", "s1", "s2", "s3", "f1", "f2", "f3", "t1", "t2"],
            )
        ) == yhat
        @test SimSpread.predict((A,B),y) == yhat
    end

    # TODO: Add unittests
    @testset "clean!" begin
        @test skip = true
    end

    # TODO: Add unittests
    @testset "save" begin
        @test skip = true
    end

end

@testset "Performance evaluation" begin
    @testset "Overall performance metrics" begin
        # TODO: Add unittests
        @testset "AuPRC" begin
            @test skip = true
        end

        # TODO: Add unittests
        @testset "AuROC" begin
            @test skip = true
        end
    end

    @testset "Early recognition metrics" begin
        # TODO: Add unittests
        @testset "BEDROC" begin
            @test skip = true
        end

        @testset "recallatL" begin
            grouping = ones(10)
            yhat = vec(collect(1:1:10))
            y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

            @test recallatL(y, yhat, grouping, 5) ≈ 3/3
            @test recallatL(y, yhat, grouping, 1) ≈ 1/3
        end

        @testset "precisionatL" begin
            grouping = ones(10)
            yhat = vec(collect(1:1:10))
            y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

            @test precisionatL(y, yhat, grouping, 5) ≈ 3/5
            @test precisionatL(y, yhat, grouping, 1) ≈ 1/1
        end
    end

    @testset "Binary prediction metrics" begin
        @testset "From confusion matrix" begin
            tn, fp, fn, tp = [3, 2, 2, 3]

            @test SimSpread.f1score(tn, fp, fn, tp) ≈ 0.6
            @test SimSpread.mcc(tn, fp, fn, tp) ≈ 0.2
            @test SimSpread.accuracy(tn, fp, fn, tp) ≈ 0.6
            @test SimSpread.balancedaccuracy(tn, fp, fn, tp) ≈ 0.6
            @test SimSpread.recall(tn, fp, fn, tp) ≈ 0.6
            @test SimSpread.precision(tn, fp, fn, tp) ≈ 0.6
        end

        @testset "From ROCNums" begin
            y, yhat = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

            @test SimSpread.f1score(roc(y, yhat)) ≈ 0.6
            @test SimSpread.mcc(roc(y, yhat)) ≈ 0.2
            @test SimSpread.accuracy(roc(y, yhat)) ≈ 0.6
            @test SimSpread.balancedaccuracy(roc(y, yhat)) ≈ 0.6
            @test SimSpread.recall(roc(y, yhat)) ≈ 0.6
            @test SimSpread.precision(roc(y, yhat)) ≈ 0.6
        end

        # TODO: Implement function
        @testset "From y and yhat vectors" begin
            y, yhat = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

            @test SimSpread.f1score(y, yhat) ≈ 0.6 skip = true
            @test SimSpread.mcc(y, yhat) ≈ 0.2 skip = true
            @test SimSpread.accuracy(y, yhat) ≈ 0.6 skip = true
            @test SimSpread.balancedaccuracy(y, yhat) ≈ 0.6 skip = true
            @test SimSpread.recall(y, yhat) ≈ 0.6 skip = true
            @test SimSpread.precision(y, yhat) ≈ 0.6 skip = true
        end

        @testset "MCC for undefined cases" begin
            yhat, y = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

            @test SimSpread.mcc(roc(y, ones(Int, 10))) - SimSpread.mcc(5, 5) < 10^-5
            @test SimSpread.mcc(roc(y, zeros(Int, 10))) - SimSpread.mcc(5, 5) < 10^-5
            @test SimSpread.mcc(roc(ones(Int, 10), yhat)) - SimSpread.mcc(5, 5) < 10^-5
            @test SimSpread.mcc(roc(zeros(Int, 10), yhat)) - SimSpread.mcc(5, 5) < 10^-5
        end
    end

    @testset "Miscellaneous metrics" begin
        # TODO: Add unittests
        @testset "meanperformance" begin
            @test skip = true
        end

        # TODO: Add unittests
        @testset "meanstdperformance" begin
            @test skip = true
        end

        # TODO: Add unittests
        @testset "maxperformanc" begin
            @test skip = true
        end
    end
end
