using Test
using SimSpread
using NamedArrays

# TODO: Add general utilities unit tests
#=
@testset "General utilities" begin
    read_namedmatrix
    k,
end
=#

@testset "prepare!" begin
    DF = NamedArray([1 0 1; 1 1 0; 0 1 1])
    DT = NamedArray([0 1; 1 1; 1 0])
    C = ["D1"]
    setnames!(DF, ["D$i" for i in 1:3], 1)
    setnames!(DF, ["fD$i" for i in 1:3], 2)
    setnames!(DT, ["D$i" for i in 1:3], 1)
    setnames!(DT, ["T$i" for i in 1:2], 2)

    A, B = prepare!(DT, DF, C)
    @test all(names(A, 1) .== names(A, 2))
    @test all(names(B, 1) .== names(B, 2))
    @test all(names(A, 1) .== ["D1", "D2", "D3", "fD2", "fD3", "T1", "T2"])
    @test all(names(B, 1) .== ["D1", "D2", "D3", "fD2", "fD3", "T1", "T2"])

    setnames!(DF, ["D$i" for i in 1:3], 2)
    @test_throws AssertionError("Features and drugs have the same names!") prepare!(DT, DF, C)
end

@testset "cutoff" begin
    x = 0.8
    y = 0.2
    z = hcat(collect(0.0:0.1:1.0))

    α = 0.5
    @test cutoff(x, α, false) ≈ 1.0
    @test cutoff(x, α, true) ≈ 0.8
    @test cutoff(y, α, false) ≈ 0.0
    @test cutoff(y, α, true) ≈ 0.0
    @test cutoff(z, α, false) ≈ [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    @test cutoff(z, α, true) ≈ [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    β = -0.01
    @test cutoff(x, β, false) ≈ 1.0
    @test cutoff(x, β, true) ≈ 0.8
    @test cutoff(y, β, false) ≈ 1.0
    @test cutoff(y, β, true) ≈ 0.2
    @test cutoff(z, β, false) ≈ ones(Float64, 11)
    @test cutoff(z, β, true) ≈ z

    γ = 1.01
    @test cutoff(x, γ, false) ≈ 0.0
    @test cutoff(x, γ, true) ≈ 0.0
    @test cutoff(y, γ, false) ≈ 0.0
    @test cutoff(y, γ, true) ≈ 0.0
    @test cutoff(z, γ, false) ≈ zeros(Float64, 11)
    @test cutoff(z, γ, true) ≈ zeros(Float64, 11)
end

# TODO: Add performance metrics unit tests
#=
@testset "Performance assessment - Metrics" begin
    y = []
    yhat = []
    @test BEDROC(y, yhat) ≈ V
    @test AuPRC(y, yhat) ≈ V
    @test AuROC(y, yhat) ≈ V
    @test f1score(y, yhat) ≈ V
    @test mcc(y, yhat) ≈ V
    @test accuracy(y, yhat) ≈ V
    @test balancedaccuracy(y, yhat) ≈ V
    @test recall(y, yhat) ≈ V
    @test precision(y, yhat) ≈ V
end

@testset "Performance assessment - Moments" begin
    y = []
    yhat = []
    @test performanceatL(y, yhat) ≈ V
    @test meanperformance(y, yhat) ≈ V
    @test meanstdperformance(y, yhat) ≈ V
    @test maxperformance(y, yhat) ≈ V
end
=#
