function plot_diff_inits(err_mean_l::AbstractVector, err_std_l::AbstractVector)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    n = length(err_mean_l)

    ind = 0:n-1  # x locations for the groups
    width = 0.8 # the width of the bars

    # the bars
    rects1 = ax.bar(ind, 100*err_mean_l, width,
                    color="cornflowerblue",
                    edgecolor = "black",
                    yerr=100*err_std_l)
                    
    ax.set_xlim(-width, length(ind))
    ax.set_xlabel("Different initializations", fontsize=17)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
    ax.set_ylabel("Error", fontsize=17)
    plt.title("Coefficient error", fontsize=17)
    plt.xticks([])
    plt.yticks(fontsize=17)
    plt.tight_layout()
    plt.savefig(joinpath("test", "plots", "diff_inits_coef_err_fput.pgf"))
    plt.close(fig)
end


function plot_diff_inits(err_l::AbstractVector)
    plot_diff_inits(err_l, zeros(length(err_l)))
end
