module experiments

using ITensors
using SparseTTRegression
using PyPlot
using BenchmarkTools
using Statistics


include("calculations.jl")
export val_diff_inits
include("plotting.jl")
export plot_diff_inits

end  # end module


# function val_diff_ODEs(Θ::MPS, Ξ::AbstractVector, dX::AbstractArray, ψ::AbstractVector, 
#     X_rec_l::AbstractVector, dX_rec_l::AbstractVector, Xi_exact::AbstractArray)
#     n = size(dX, 2)
#     iter_rec = length(X_rec_l)

#     rel_train_err_l = []
#     rec_err_mean_l = []
#     rec_err_std_l = []
#     Theta_rec_l = []
#     coef_err_l = []

#     for i in range(1, iter_rec)
#         Θ_rec = coordinate_major(X_rec_l[i], ψ)
#         push!(Theta_rec_l, Matrix(Θ_rec))
#     end

#     for q in range(1, length(Ξ))
#         rel_train_err = norm(Matrix(Θ)*Matrix(Ξ[q]) - dX[:, q])/norm(dX[:, q])*100
#         push!(rel_train_err_l, rel_train_err)

#         # reconstruct derivative for validation
#         rec_err_l = []
#         for i in range(1, iter_rec)
#             rec_err = norm(Theta_rec_l[i]*Matrix(Ξ[q]) - dX_rec_l[i][:, q])/norm(dX_rec_l[i][:, q])*100
#             push!(rec_err_l, rec_err)
#         end
#         coef_err = norm(Matrix(Ξ[q]) - Xi_exact[:, q])/norm(Xi_exact[:, q])*100
#         push!(coef_err_l, coef_err)
#         mean_val_err = mean(rec_err_l)
#         std_val_err = std(rec_err_l)
#         push!(rec_err_mean_l, mean_val_err)
#         push!(rec_err_std_l, std_val_err)
#     end
#     return rec_err_mean_l, rec_err_std_l, coef_err_l
# end


# function plot_diff_ODEs(err_mean_l, err_std_l)
#     fig1 = plt.figure()
#     ax1 = fig1.add_subplot(111)

#     n = length(err_mean_l)

#     ind = 1:n  # x locations for the groups
#     width = 0.8 # the width of the bars

#     # the bars
#     rects1 = ax1.bar(ind, err_mean_l, width,
#                     color="cornflowerblue",
#                     edgecolor = "black",
#                     yerr=err_std_l)
                    
#     ax1.set_xlim(0.15, length(ind)+width)
#     ax1.set_xlabel("ODE number", fontsize=17)
#     ax1.set_ylabel("Error", fontsize=17)
#     ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.title("Reconstruction error", fontsize=17)
#     ax1.set_xticks(ind)
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "diff_ODEs_rec_err_test_model.pgf"))
#     plt.close(fig1)
# end


# function plot_diff_ODEs(err_l)
#     plot_diff_ODEs(err_l, zeros(length(err_l)))
# end



# # different methods in order with legend names, marker, color, rgb
# # MALS LASSO: ".", blue, 0.121569,0.466667,0.705882
# # LASSO: "x", red, 0.839216,0.152941,0.156863
# # MANDy: ">", orange, 1.000000,0.498039,0.054902
# # SINDy: ".", green, 0.172549,0.627451,0.172549
# function val_time_m(X_train_l::AbstractVector, ddX_train_l::AbstractVector, ψ::AbstractVector, λ::AbstractVector)
#     n_train = length(X_train_l)

#     number_timesteps = []
#     for i in range(1, n_train)
#         push!(number_timesteps, size(X_train_l[i], 1))
#     end

#     suite = BenchmarkGroup()
#     suite["regression"] = BenchmarkGroup()
#     for i in range(1, n_train)
#         ΘΘ = coordinate_major(X_train_l[i], ψ)
#         m = number_timesteps[i]
#         ddX = ddX_train_l[i]
#         while true
#             try
#                 suite["regression"]["als", m] = @benchmark sparse_TT_optimization($ΘΘ, $ddX,  $λ, sweep_count=1)
#                 break
#             catch
#                 suite["regression"]["als", m] = NaN
#                 break
#             end
#         end
#         while true
#             try
#                 suite["regression"]["mandy", m] = @benchmark mandy($ΘΘ, $ddX)
#                 break
#             catch
#                 suite["regression"]["mandy", m] = NaN
#                 break
#             end
#         end
#         while true
#             try
#                 suite["regression"]["lasso", m] = @benchmark regular_lasso($(Matrix(ΘΘ)), $ddX, $λ)
#                 break
#             catch
#                 suite["regression"]["lasso", m] = NaN
#                 break
#             end
#         end
#         while true
#             try
#                 suite["regression"]["sindy", m] = @benchmark sindy($(Matrix(ΘΘ)), $ddX, $0.05, $10)
#                 break
#             catch
#                 suite["regression"]["sindy", m] = NaN
#                 break
#             end
#         end
#     end
#     return suite
# end


# function val_time_n(X_train_l::AbstractVector, ddX_train_l::AbstractVector, ψ::AbstractVector, λ::AbstractVector)
#     n_train = length(X_train_l)

#     n_ODEs_l = []
#     for i in range(1, n_train)
#         push!(n_ODEs_l, size(X_train_l[i], 2))
#     end

#     suite = BenchmarkGroup()
#     suite["regression"] = BenchmarkGroup()
#     for i in range(1, n_train)
#         ΘΘ = coordinate_major(X_train_l[i], ψ)
#         n = n_ODEs_l[i]
#         ddX = ddX_train_l[i]
#         while true
#             try
#                 suite["regression"]["als", n] = @benchmark sparse_TT_optimization($ΘΘ, $ddX,  $λ, sweep_count=1)
#                 break
#             catch
#                 suite["regression"]["als", n] = NaN
#                 break
#             end
#         end
#         while true
#             try
#                 suite["regression"]["mandy", n] = @benchmark mandy($ΘΘ, $ddX)
#                 break
#             catch
#                 suite["regression"]["mandy", n] = NaN
#                 break
#             end
#         end
#         while true
#             try
#                 suite["regression"]["lasso", n] = @benchmark regular_lasso($(Matrix(ΘΘ)), $ddX, $λ)
#                 break
#             catch
#                 suite["regression"]["lasso", n] = NaN
#                 break
#             end
#         end
#         while true
#             try
#                 suite["regression"]["sindy", n] = @benchmark sindy($(Matrix(ΘΘ)), $ddX, $0.05, $10)
#                 break
#             catch
#                 suite["regression"]["sindy", n] = NaN
#                 break
#             end
#         end
#     end

#     # calculate medians
#     med_als = [isa(suite["regression"]["als", n], Number) ? NaN : median(suite["regression"]["als", n]).time for n in n_ODEs_l].* 10^-6
#     med_mandy = [isa(suite["regression"]["mandy", n], Number) ? NaN : median(suite["regression"]["mandy", n]).time for n in n_ODEs_l].* 10^-6
#     med_sindy = [isa(suite["regression"]["sindy", n], Number) ? NaN : median(suite["regression"]["sindy", n]).time for n in n_ODEs_l].* 10^-6
#     med_lasso = [isa(suite["regression"]["lasso", n], Number) ? NaN : median(suite["regression"]["lasso", n]).time for n in n_ODEs_l].* 10^-6

#     # plot
#     fig = plt.figure()
#     plt.plot(n_ODEs_l, med_als, label="MALS LASSO", linewidth=3, marker="o", markersize=13)
#     plt.plot(n_ODEs_l, med_mandy, label="MANDy", linewidth=3, marker="+", markersize=13)
#     plt.plot(n_ODEs_l, med_sindy, label="SINDy", linewidth=3, marker="x", markersize=13)
#     plt.plot(n_ODEs_l, med_lasso, label="Regular LASSO", linewidth=3, marker=">", markersize=13)

#     plt.xlabel("n", fontsize=17)
#     plt.ylabel("Computational cost in ms", fontsize=17)
#     # plt.title("Different number of ODEs", fontsize=17)
#     plt.legend(fontsize=17)
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "time_n_fput"))
#     plt.close(fig)
#     return suite
# end





# function val_time_n_trunc(X_train_l::AbstractVector, ddX_train_l::AbstractVector, ψ::AbstractVector, 
#     truncations::AbstractVector, λ::AbstractVector)
#     n_train = length(X_train_l)

#     n_ODEs_l = []
#     for i in range(1, n_train)
#         push!(n_ODEs_l, size(X_train_l[i], 2))
#     end

#     suite = BenchmarkGroup()
#     suite["truncation"] = BenchmarkGroup()
#     for i in range(1, n_train)
#         ΘΘ = coordinate_major(X_train_l[i], ψ)
#         n = n_ODEs_l[i]
#         ddX = ddX_train_l[i]

#         for trunc in truncations
#             while true
#                 try
#                     suite["truncation"][trunc, n] = @benchmark truncated_TToptimization($ΘΘ, $ddX, $λ, $trunc)
#                     break
#                 catch
#                     suite["truncation"][trunc, n] = NaN
#                     break
#                 end
#             end
#         end
#     end

#     # calculate medians
#     med_l = []
#     for trunc in truncations
#         med = [isa(suite["truncation"][trunc, n], Number) ? NaN : median(suite["truncation"][trunc, n]).time for n in n_ODEs_l].* 10^-6
#         push!(med_l, med)
#     end
    
#     # plot
#     fig = plt.figure()
#     for (i, trunc) in enumerate(truncations)
#         if trunc == 0
#             plt.plot(n_ODEs_l, med_l[i], label="tol = "*string(trunc), linewidth=3, marker="o", markersize=13)
#         else
#             plt.plot(n_ODEs_l, med_l[i], label="tol = "*@sprintf("%.0e", trunc), linewidth=3, marker="o", markersize=13)
#         end
#     end

#     plt.xlabel("n", fontsize=17)
#     plt.ylabel("Computational cost in ms", fontsize=17)
#     plt.legend(fontsize=17)
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "time_n_fput_truncations"))
#     plt.close(fig)
#     return suite
# end


# function val_diff_noise(X_train_l::AbstractVector, ddX_train_l::AbstractVector, ψ::AbstractVector, λ::AbstractVector,
#     X_test_l::AbstractVector, ddX_test_l::AbstractVector, regression_method::Function; error_func=rec::Function)
    
#     rec_err_l = Float64[]
#     train_err_l = Float64[]

#     for i in range(1, length(X_train_l))
#         print("calculating noise level", i,"\n")
#         Θ = coordinate_major(X_train_l[i], ψ)
#         _, train_err, rec_err = regression_method(Θ, ddX_train_l[i], ψ, λ, X_test_l, ddX_test_l; error_func=error_func)
#         push!(rec_err_l, rec_err)
#         push!(train_err_l, train_err)
#     end
#     return train_err_l, rec_err_l
# end


# # MALS LASSO: ".", blue, 0.121569,0.466667,0.705882
# # LASSO: "x", red, 0.839216,0.152941,0.156863
# # MANDy: "+", orange, 1.000000,0.498039,0.054902
# # SINDy: ">", green, 0.172549,0.627451,0.172549
# function plot_diff_noise(noise_level::AbstractVector, diff_noise_results::Dict)
#     colors = [(0.121569,0.466667,0.705882), (0.839216,0.152941,0.156863), 
#     (1.000000,0.498039,0.054902), (0.172549,0.627451,0.172549)]
#     markers = [".", "x", "+", ">"]
#     order = [0, 5, 10, 15]
#     styles = ["--", "-"]  # linestyle for: -- train error, - rec error
#     err_names = ["Train error", "Reconstruction error"]

#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)

#     for (i, key) in enumerate(permute!(collect(keys(diff_noise_results)), [2,1,3,4]))
#         # ax1.plot(noise_level, 100*diff_noise_results[key][1], 
#         # linewidth=2, c=colors[i], linestyle="--", marker=markers[i], markersize=10, zorder=order[i])
#         ax1.plot(noise_level, 100*diff_noise_results[key][2], label=key,
#         linewidth=2, c=colors[i], marker=markers[i], markersize=10, zorder=order[i])
#     end

#     plt.xlabel("\$ \\sigma^2\$", fontsize=17)
#     plt.ylabel("Error", fontsize=17)
#     plt.title("Reconstruction error", fontsize=17)
#     # plt.yscale("log")
#     ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     ax1.set_xticks([0, 0.04, 0.08, 0.12, 0.16, 0.2])
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)

#     # ax2 = ax1.twinx()
#     # for (i, style) in enumerate(styles)
#     #     ax2.plot(NaN, NaN, ls=style, label= err_names[i], c="black")
#     # end
#     # ax2.get_yaxis().set_visible(false)
#     # ax2.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0, 0.8))
#     ax1.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.24, 0.69))
#     # ax2.legend(fontsize=12, loc="upper right")
#     # ax1.legend(fontsize=12, loc="center right")

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "diff_noise_rec.pgf"))
#     plt.close(fig)
# end




# function val_learning_curve(X_train_l::AbstractVector, ddX_train_l::AbstractVector, ψ::AbstractVector, λ::AbstractVector,
#     X_test_l::AbstractVector, ddX_test_l::AbstractVector, regression_method::Function; error_func=rec::Function)
    
#     rec_err_l = Float64[]
#     train_err_l = Float64[]

#     for i in range(1, length(X_train_l))
#         print("calculating m = ", size(X_train_l[i], 1), "\n")
#         Θ = coordinate_major(X_train_l[i], ψ)
#         _, train_err, rec_err = regression_method(Θ, ddX_train_l[i], ψ, λ, X_test_l, ddX_test_l; error_func=error_func)
#         push!(rec_err_l, rec_err)
#         push!(train_err_l, train_err)
#     end
#     return train_err_l, rec_err_l
# end


# function plot_learning_curve(timesteps::AbstractVector, learning_curve_results::Dict)
#     # colors = [(0.121569,0.466667,0.705882), (0, 136, 116)./255, (0.839216, 0.152941, 0.156863)]
#     # colors = [(0.121569,0.466667,0.705882), (190, 79, 48)./255, (0, 136, 116)./255]
#     colors = [(1.000000,0.498039,0.054902), (84, 67, 58)./255, (0, 139, 111)./255]
#     markers = ["+", "^", "^", ">", "*"]
#     order = [10, 5, 0]
#     styles = ["--", "-"]  # linestyle for: -- train error, - rec error
#     err_names = ["Train error", "Reconstruction error"]

#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)

#     for (i, key) in enumerate(permute!(collect(keys(learning_curve_results)), [2,1]))
#         ax1.plot(timesteps, 100*learning_curve_results[key][1], 
#         linewidth=2, c=colors[i], linestyle="--", marker=markers[i], markersize=10, zorder=order[i])
#         ax1.plot(timesteps, 100*learning_curve_results[key][2], label=key,
#         linewidth=2, c=colors[i], marker=markers[i], markersize=10, zorder=order[i])
#     end

#     # for (i, method) in enumerate(learning_curve_results)
#     #     ax1.plot(timesteps[2:end], 100*method[2][1][2:end], 
#     #     linewidth=2, c=colors[i], linestyle="--", marker=markers[i], markersize=10)
#     #     ax1.plot(timesteps[2:end], 100*method[2][2][2:end], label=method[1],
#     #     linewidth=2, c=colors[i], marker=markers[i], markersize=10)
#     # end

#     plt.xlabel("m", fontsize=17)
#     plt.ylabel("Error", fontsize=17)
#     plt.title("MANDy", fontsize=17)
#     # plt.yscale("log")
#     ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)

#     ax2 = ax1.twinx()
#     for (i, style) in enumerate(styles)
#         ax2.plot(NaN, NaN, ls=style, label= err_names[i], c="black")
#     end
#     ax2.get_yaxis().set_visible(false)
#     ax2.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0, 0.8))
#     ax1.legend(fontsize=12, loc="upper left")
#     # ax2.legend(fontsize=12, loc="upper right")
#     # ax1.legend(fontsize=12, loc="center right")

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "learning_curve_n8_mandy_noisy.pgf"))
#     plt.close(fig)
# end


# function plot_difference_lasso(timesteps::AbstractVector, difference::AbstractVector)
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)

#     ax.plot(timesteps, 100*difference, 
#     linewidth=2, c="purple", linestyle="-", marker=".", markersize=10, label="\$err_{MALS\\,LASSO} - err_{LASSO}\$")


#     plt.xlabel("m", fontsize=17)
#     plt.ylabel("Difference error", fontsize=17)
#     plt.title("Noisy", fontsize=17)
#     ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     ax.legend(fontsize=12, loc="upper right")

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "learning_curve_n8_difference_lasso_noisy.pgf"))
#     plt.close(fig)
# end


# function val_err_diff_n(ψ::AbstractVector, n_ODEs::AbstractVector, λ::AbstractVector, 
#     regression_method::Function; error_func=rec::Function)
#     train_err_l = Float64[]
#     rec_err_l = Float64[]
#     for n in n_ODEs
#         X_train = Matrix(CSV.read("test\\data\\fput\\reconstruction_error_different_n\\"*string(n)*"\\X_train_noisy.csv", DataFrame))
#         # ddX_train = Matrix(CSV.read("test\\data\\fput\\reconstruction_error_different_n\\"*string(n)*"\\ddX_train_noisy.csv", DataFrame))[:, 2:end-1]
#         ddX_train = call_derivative_fput(X_train, 0.25)[1:400, 2:end-1]
#         X_train = X_train[1:400, 2:end-1]
#         X_test_l = []
#         ddX_test_l = []

#         for i in range(1,9) 
#             push!(X_test_l, Matrix(CSV.read("test\\data\\fput\\reconstruction_error_different_n\\"*string(n)*"\\X_test"*string(i)*".csv", DataFrame))[:, 2:end-1])
#             push!(ddX_test_l, Matrix(CSV.read("test\\data\\fput\\\\reconstruction_error_different_n\\"*string(n)*"\\ddX_test"*string(i)*".csv", DataFrame)[:, 2:end-1]))
#         end

#         print("Calculate n = ", n, "\n")
#         Θ = coordinate_major(X_train, ψ)
#         _, train_err, rec_err = regression_method(Θ, ddX_train, ψ, λ, X_test_l, ddX_test_l; 
#         error_func=error_func)
#         push!(train_err_l, train_err)
#         push!(rec_err_l, rec_err)
#     end
#     return train_err_l, rec_err_l
# end


# function plot_err_diff_n(n_ODEs::AbstractVector, err_diff_n::Dict)

#     colors = [(0.121569,0.466667,0.705882), (0.839216,0.152941,0.156863), "red", "g"]  # in this order the colors of: MALS LASSO, MANDy, LASSO, SINDy
#     markers = [".", "x", "+", ">"]
#     styles = ["--", "-"]  # linestyle for: -- train error, - rec error
#     err_names = ["Train error", "Reconstruction error"]
#     order = [0,5,25,15,20,10]
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)
#     # for (i, method) in enumerate(err_diff_n)
#     #     ax1.plot(n_ODEs, 100*method[2][1], 
#     #     linewidth=2, c=colors[i], linestyle="--", marker=markers[i], markersize=10)
#     #     ax1.plot(n_ODEs, 100*method[2][2], label=method[1],
#     #     linewidth=2, c=colors[i], marker=markers[i], markersize=10)
#     # end
#     for (i, key) in enumerate(permute!(collect(keys(err_diff_n)), [2,1]))
#         ax1.plot(n_ODEs, 100*err_diff_n[key][1], 
#         linewidth=2, c=colors[i], linestyle="--", marker=markers[i], markersize=10, zorder=order[i])
#         ax1.plot(n_ODEs, 100*err_diff_n[key][2], label=key,
#         linewidth=2, c=colors[i], marker=markers[i], markersize=10, zorder=order[i])
#     end
    
#     plt.xlabel("n", fontsize=17)
#     plt.ylabel("Error", fontsize=17)
#     plt.title("", fontsize=17)

#     ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)

#     ax2 = ax1.twinx()
#     for (i, style) in enumerate(styles)
#         ax2.plot(NaN, NaN, ls=style, label= err_names[i], c="black")
#     end
#     ax2.get_yaxis().set_visible(false)

#     ax2.legend(fontsize=12, loc="upper right")
#     ax1.legend(fontsize=12, loc="upper right", bbox_to_anchor=(1,0.8))
#     # ax2.legend(loc="upper left", fontsize=12)
#     # ax1.legend(bbox_to_anchor=(0.41, 0.5), fontsize=11)

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "diff_n_m400_mals_lasso_vs_lasso.pgf"))
#     plt.close(fig)
# end


# function plot_diff_n_diff_m(n_ODEs::AbstractVector, err_diff_n::Dict)
#     colors = [(0.121569,0.466667,0.705882), (0.839216,0.152941,0.156863), "purple", "green", "orange"]  # in this order the colors of: MALS LASSO, MANDy, LASSO, SINDy
#     markers = [".", "x", "+", ">", "x"]
#     styles = ["-"]  # linestyle for: -- train error, - rec error
#     err_names = ["Reconstruction error"]
#     order = [0,5,25,15,20,10]
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)
#     for (i, key) in enumerate(permute!(collect(keys(err_diff_n)), [2,1]))

#         ax1.plot(n_ODEs, 100*err_diff_n[key][2], label=key,
#         linewidth=2, c=colors[i], marker=markers[i], markersize=10, zorder=order[i])
#     end

#     plt.xlabel("n", fontsize=17)
#     plt.ylabel("Error", fontsize=17)
#     plt.title("MALS LASSO", fontsize=17)

#     ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     ax2 = ax1.twinx()
#     for (i, style) in enumerate(styles)
#         ax2.plot(NaN, NaN, ls=style, label= err_names[i], c="black")
#     end
#     ax2.get_yaxis().set_visible(false)

#     ax2.legend(fontsize=12, loc="upper right")
#     ax1.legend(fontsize=12, loc="upper right", bbox_to_anchor=(1,0.9))

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "diff_n_m400_mals_lasso.pgf"))
#     plt.close(fig)
# end

# function plot_diff_n_difference_lasso(n_ODEs::AbstractVector, difference::AbstractVector)
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)

#     ax.plot(n_ODEs, 100*difference, 
#     linewidth=2, c=(0.121569,0.466667,0.705882), linestyle="-", marker=".", markersize=10, label="\$err_{MALS\\,LASSO} - err_{LASSO}\$")


#     plt.xlabel("n", fontsize=17)
#     plt.ylabel("Difference error", fontsize=17)
#     plt.title("Difference to LASSO", fontsize=17)
#     ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     ax.legend(fontsize=12, loc="upper right")

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "diff_n_m400_mals_lasso.png"))
#     plt.close(fig)
# end

# function call_derivative_fput(X::AbstractMatrix{T}, β::Number) where{T}
#     m, n = size(X)
#     ddX = zeros(T, m, n)
#     for i in range(1, m)
#         for j in 2:n-1
#             ddX[i, j] = X[i, j+1] - 2*X[i, j] + X[i, j-1] + β*((X[i, j+1] - X[i, j])^2 - (X[i, j] - X[i, j-1])^2)
#         end
#     end
#     return ddX
# end


# function val_trunc(truncations::AbstractVector, X_train::AbstractArray, ddX_train::AbstractArray, ψ::AbstractVector, λ::AbstractVector,
#     X_test_l::AbstractVector, ddX_test_l::AbstractVector, regression_method::Function; error_func=rec::Function)
    
#     rec_err_l = Float64[]
#     train_err_l = Float64[]
    
#     for (i, trunc) in enumerate(truncations)
#         print("calculating truncation ", i,"\n")
#         Θ = coordinate_major(X_train, ψ)
#         truncate!(Θ, cutoff = trunc)
#         _, train_err, rec_err = regression_method(Θ, ddX_train, ψ, λ, X_test_l, ddX_test_l; error_func=error_func)
        
#         push!(rec_err_l, rec_err)
#         push!(train_err_l, train_err)
#     end
#     return train_err_l, rec_err_l
# end


# function plot_trunc(truncations::AbstractVector, diff_trunc_results::Dict)

#     colors = 
#     [(0.121569, 0.466667, 0.705882), 
#     (1.000000,0.498039,0.054902),
#     (0.839216,0.152941,0.156863), 
#     (0.172549,0.627451,0.172549)]  # in this order the colors of: MALS LASSO, MANDy, LASSO, SINDy
#     markers = [".", "+", "x", ">"]  # in this order the markers of: MALS LASSO, MANDy, LASSO, SINDy
#     styles = ["--", "-"]  # linestyle for: -- train error, - rec error
#     err_names = ["Train error", "Reconstruction error"]
#     order = [0,5,10,15]
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1,1,1)
#     for (i, key) in enumerate(permute!(collect(keys(diff_trunc_results)), [1,2,3,4]))
#         ax1.plot(truncations, 100*diff_trunc_results[key][1], 
#         linewidth=2, c=colors[i], linestyle="--", marker=markers[i], markersize=10, zorder=order[i])
#         ax1.plot(truncations, 100*diff_trunc_results[key][2], label=key,
#         linewidth=2, c=colors[i], marker=markers[i], markersize=10, zorder=order[i])
#     end


#     plt.xlabel("tol", fontsize=17)
#     plt.ylabel("Error", fontsize=17)
#     plt.title("\$ n = 16 \$", fontsize=17)

#     ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
#     plt.xticks(fontsize=17)
#     plt.yticks(fontsize=17)
#     plt.xscale("log")

#     ax2 = ax1.twinx()
#     for (i, style) in enumerate(styles)
#         ax2.plot(NaN, NaN, ls=style, label= err_names[i], c="black")
#     end
#     ax2.get_yaxis().set_visible(false)

#     ax2.legend(fontsize=12, loc="upper left")
#     ax1.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0.0,0.62))

#     plt.tight_layout()
#     plt.savefig(joinpath("test", "plots", "diff_trunc_n16.png"))
#     plt.savefig(joinpath("test", "plots", "diff_trunc_n16.pgf"))
#     plt.close(fig)
# end