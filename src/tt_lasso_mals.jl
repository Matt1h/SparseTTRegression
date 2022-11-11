function mals_lasso(tt_opt::MPS, A::MPS, b::AbstractVector, i::Integer, λ::AbstractVector;
	tol=1e-6::Number, svd_tol=1e-14::Number, lasso_tol=1e-7::Number,
	verbose=false::Bool, full=false::Bool, make_plot=false::Bool, sweep_count=1::Integer)

    ord = length(tt_opt)
	
	orthogonalize!(tt_opt, 1)

	# initialize G
	G = repeat([ITensor()], ord-1)
	G[1] = A[1]

	# initialize H
	H = init_H_mals(tt_opt, A)
	
	trace_forward_full = []
	trace_backward_full = []
	trace_forward = []
	trace_backward = []

	n_sweeps = 0  # sweeps counter
	converged = false

	while true
		n_sweeps+=1
		# First half sweep

		for q = 1:(ord-1)
			verbose ? println("─Forward sweep: core optimization $q out of ", ord-1,) : nothing

			# Lasso optimization of core q
			Aq, V_inds = Aq_full(G[q], H[q])  # TODO: cotrol ranksize
			f = fit(LassoPath, Aq, b, λ=λ, maxncoef=ITensors.dim(V_inds), 
			cd_tol=lasso_tol, standardize = false, intercept = false)

			# reshape and insert updated core
			V_m = Matrix(f.coefs)[:, length(λ)]
			V = ITensor(V_m, V_inds)
			tt_opt = right_core_move_mals(tt_opt, V, q, svd_tol)

			q < ord-1 ? G[q+1] = G[q]*tt_opt[q]*A[q+1] : nothing  
		end

		n_sweeps+=1
		# Second half sweep
		for q = ord-1:(-1):1
			verbose ? println("─Bakward sweep: core optimization $q out of ", ord-1,) : nothing

			# Lasso optimization of core q
			Aq, V_inds = Aq_full(G[q], H[q])  # TODO: cotrol ranksize
			f = fit(LassoPath, Aq, b, λ=λ, maxncoef=ITensors.dim(V_inds),
			cd_tol=lasso_tol, standardize = false, intercept = false)

			# reshape and insert updated core
			V_m = Matrix(f.coefs)[:, length(λ)]
			V = ITensor(V_m, V_inds)
			tt_opt = left_core_move_mals(tt_opt, V, q, svd_tol)

			q > 1 ? H[q-1] = tt_opt[q+1]*A[q]*H[q] : nothing
		end

		if n_sweeps >= sweep_count && !converged
			trace_forward = trace_forward_full
			trace_backward = trace_backward_full
			break
		elseif n_sweeps >= sweep_count && converged
			break
		end

	end
	return tt_opt
end



function init_H_mals(U::MPS, A::MPS)
	ord = length(U)
	H = Array{ITensor}(undef, ord-1)
	H[ord-1] = A[ord+1]*A[ord]
	for q = ord-1:-1:2
		H[q-1] = U[q+1]*A[q]*H[q]  # update H
	end
	return H
end


function init_H_u_mals(U::MPS)
	ord = length(U)
	H_u = Array{Union{ITensor, Nothing}}(undef, ord-1)
	H_u[ord-1] = nothing
	H_u[ord-2] = U[end]
	for q = ord-2:-1:2
		H_u[q-1] = U[q+1]*H_u[q]  # update H
	end
	return H_u
end


function Aq_full(Gq::ITensor, Hq::ITensor)
	Aq = Gq*Hq
	Aq_inds = inds(Aq)

	ColInd = filter(x -> hastags(x, "ColDim"), Aq_inds)
	V_inds = filter(x -> !hastags(x, "ColDim"), Aq_inds)

	col_loc = findall(x -> hastags(x, "ColDim"), Aq_inds)
	non_col_loc = findall(x -> !hastags(x, "ColDim"), Aq_inds)

	Aq_array = permutedims(Array(Aq, Aq_inds), [col_loc; non_col_loc])
	return reshape(Aq_array, ITensors.dim(ColInd), ITensors.dim(V_inds)), V_inds
end


function right_core_move_mals(U::MPS, V::ITensor, q::Integer, tol::Number)
	if ITensors.order(V) == 3 && q == 1
		u, s, v = svd(V, inds(V)[1], cutoff=tol)  # TODO: maxdim instead of cutoff also possible
	elseif ITensors.order(V) == 4 || (ITensors.order(V) == 3 && q > 1)
		u, s, v = svd(V, inds(V)[1:2], cutoff=tol)
	end
	U[q] = u
	U[q+1] = s*v
	return U
end


function left_core_move_mals(U::MPS, V::ITensor, q::Integer, tol::Number)
	if ITensors.order(V) == 3 && q == 1
		u, s, v = svd(V, inds(V)[1], cutoff=tol)  # TODO: maxdim instead of cutoff also possible
	elseif ITensors.order(V) == 4 || (ITensors.order(V) == 3 && q > 1)
		u, s, v = svd(V, inds(V)[1:2], cutoff=tol)
	end
	U[q+1] = v
	U[q] = u*s
	return U
end


function cost_function(Gq::ITensor, Hq::ITensor, V::ITensor, 
	G_uq::Union{Nothing, ITensor}, H_uq::Union{Nothing, ITensor}, b::AbstractVector, λ::Number)
	if G_uq != nothing && H_uq != nothing
		U = G_uq*V*H_uq
	elseif  G_uq == nothing && H_uq != nothing
		U = V*H_uq
	elseif  G_uq != nothing && H_uq == nothing
		U = G_uq*V
	else
		U = V
	end
	
	K = Gq*Hq*V
	return sqrt(sum(((K - ITensor(b, inds(K))).^2).tensor)) + λ*sum(abs.(U.tensor))
end


function convergence_check(trace_forward_full::AbstractVector, trace_backward_full::AbstractVector, tol::Number)
	return norm(trace_forward_full[end] - trace_backward_full[end]) < tol || trace_forward_full[end] < trace_backward_full[end]
end


function optimal_tt(tt_opt_forward::AbstractVector, tt_opt_backward::AbstractVector, 
	trace_forward_full::AbstractVector, trace_backward_full::AbstractVector)

	ord = length(tt_opt_forward[1])
	
	minimum_forward, loc_forward = findmin(trace_forward_full)
	minimum_backward, loc_backward = findmin(trace_backward_full)
	if minimum_forward < minimum_backward
		tt_opt = tt_opt_forward[loc_forward]
		trace_forward = trace_forward_full[1:loc_forward]
		trace_backward = trace_backward_full[1:end-ord-1]
	else
		tt_opt = tt_opt_backward[loc_backward]
		trace_forward = deepcopy(trace_forward_full)
		trace_backward = trace_backward_full[1:loc_backward]
	end
	return tt_opt, trace_forward, trace_backward
end


function plot_trace(trace_forward::AbstractVector, trace_forward_full::AbstractVector,
    trace_backward::AbstractVector, trace_backward_full::AbstractVector,
    n_core_opt::Integer, name::AbstractString; full=false)

    x_forward, x_backward = trace_x_values(trace_forward, trace_backward, n_core_opt)

    Plots.scatter(x_forward, trace_forward, lab="Forward sweep", markercolor="blue")
    Plots.scatter!(x_backward, trace_backward, lab="Backward sweep", markercolor="red")

    n_core_forward = length(trace_forward)
    n_core_backward = length(trace_backward)
    if full
        x_forward, x_backward = trace_x_values(trace_forward_full, trace_backward_full, n_core_opt)
    
        Plots.scatter!(x_forward[n_core_forward+1:end], trace_forward_full[n_core_forward+1:end], lab="Forward sweep after convergence", markeralpha=0.5, markercolor="blue")
        Plots.scatter!(x_backward[n_core_backward+1:end], trace_backward_full[n_core_backward+1:end], lab="Backward sweep after convergence", markeralpha=0.5, markercolor="red")
    end
    xlabel!("Number of sweeps")
    ylabel!("Lasso cost function")
    Plots.savefig(name*".png")
end


function trace_x_values(trace_forward::AbstractVector, trace_backward::AbstractVector, n_core_opt::Integer)
    step = 1/(2*n_core_opt)
    
    n_sweeps_forward = floor(Int, length(trace_forward)/n_core_opt)
    n_core_extra_forward = mod(length(trace_forward), n_core_opt)
    n_sweeps_backward = floor(Int, length(trace_backward)/n_core_opt)
    n_core_extra_backward = mod(length(trace_backward), n_core_opt)

    x_forward = vcat([collect(i+step:step:i+1/2) for i in range(0, n_sweeps_forward-1)]...)
    n_core_extra_forward != 0 ? x_forward = [x_forward; collect(n_sweeps_forward+step:step:n_sweeps_forward + step*n_core_extra_forward)] : nothing
    x_backward = vcat([collect(i+1/2+step:step:i+1) for i in range(0, n_sweeps_backward-1)]...)
    n_core_extra_backward != 0 ? x_backward = [x_backward; collect(n_sweeps_backward+1/2+step:step:n_sweeps_backward+1/2 + step*n_core_extra_backward)] : nothing
    return x_forward, x_backward
end



# function explicit_sum(V::ITensor, G_uq::Union{Nothing, ITensor}, H_uq::Union{Nothing, ITensor})
# 	if G_uq != nothing && H_uq != nothing
# 		U = G_uq*V*H_uq
# 	elseif  G_uq == nothing && H_uq != nothing
# 		U = V*H_uq
# 	elseif  G_uq != nothing && H_uq == nothing
# 		U = G_uq*V
# 	else
# 		U = V
# 	end

# 	return sum(abs.(U.tensor))
# end


# function core_sum(A::MPS)
#     s = 0
#     for q in range(1, length(A))                                                                                                               
#         s += sum(abs.(A[q].tensor))                                                                                                                      
#     end
#     return s
# end


# function plot_info_criteria(λ::AbstractVector, V_m::AbstractMatrix, tt_opt::MPS, V_inds, 
# 	G_uq::Union{Nothing, ITensor}, H_uq::Union{Nothing, ITensor}, svd_tol::Number, q::Integer, name::AbstractString)
# 	core_sum_l = []
# 	explicit_sum_l1 = []
# 	explicit_sum_l2 = []
# 	for i in range(1, length(λ))
# 		V = ITensor(V_m[:, i], V_inds)
# 		tt_opt = right_core_move_mals(tt_opt, V, q, svd_tol)
# 		push!(core_sum_l, core_sum(tt_opt))
# 		push!(explicit_sum_l1, explicit_sum(V, G_uq, H_uq))
# 		push!(explicit_sum_l2, sum(abs.(Matrix(tt_opt))))
# 	end
# 	fig = plt.figure()
# 	plt.plot(λ, core_sum_l, label="Core sum", linewidth=3, marker="o", markersize=13)
#     plt.plot(λ, explicit_sum_l1, label="Explicit sum", linewidth=3, marker="+", markersize=13)
# 	plt.plot(λ, explicit_sum_l2, label="Explicit sum", linewidth=3, marker="+", markersize=13)
# 	plt.xlabel("lambda", fontsize=17)
#     plt.ylabel("Sum", fontsize=17)
#     plt.legend(fontsize=17)
#     plt.xticks(fontsize=17)
# 	plt.xscale("log")
#     plt.yticks(fontsize=17)
#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "sum", name))
#     plt.close(fig)
# end


# function cost_function(Gq::ITensor, Hq::ITensor, V::ITensor, U::MPS, b::AbstractVector, λ::Number)
# 	K = Gq*Hq*V
# 	return sqrt(sum((K - ITensor(b, inds(K))).^2)) + λ*sum(abs.(Matrix(U)))
# end