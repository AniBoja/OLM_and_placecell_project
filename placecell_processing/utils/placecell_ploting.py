import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


def group_scatter_bar(
    group1, group2, x_labels, y_label, group_labels, title, z_value=True
):

    if z_value:
        print("ztrue")
        group1_z_scores = [np.arctanh(coors) for coors in group1]

        group2_z_scores = [np.arctanh(coors) for coors in group2]
    else:
        print("zfalse")
        group1_z_scores = group1
        group2_z_scores = group2

    p_values = []
    t_values = []

    group1_color = "blue"
    group2_color = "lightblue"

    # create plot for ratemaps
    group_bar_scatter = go.Figure()

    for i in range(len(group1)):

        group1_test = group1_z_scores[i][~np.isnan(group1_z_scores[i])]
        group2_test = group2_z_scores[i][~np.isnan(group2_z_scores[i])]

        t_val, p_val = stats.ttest_ind(group1_test, group2_test)

        p_values.append(p_val)
        t_values.append(t_val)

    group_bar_scatter.add_trace(
        go.Bar(
            x=x_labels,
            y=[np.nanmean(vals) for vals in group1],
            marker_color=group1_color,
            name=group_labels[0],
        )
    )

    group_bar_scatter.add_trace(
        go.Bar(
            x=x_labels,
            y=[np.nanmean(vals) for vals in group2],
            marker_color=group2_color,
            name=group_labels[1],
        )
    )

    group_bar_scatter.layout.xaxis2 = go.layout.XAxis(
        overlaying="x", range=[0, len(group1)], showticklabels=False
    )
    bargap = 0.2

    for i in range(len(group1)):

        x = [i + bargap / 2 + (1 - bargap) / 4, i + 1 - bargap / 2 - (1 - bargap) / 4]
        # x = [0.5+bargap/2, 1.5-bargap/2]

        for combination in itertools.zip_longest(group1[i], group2[i]):
            y = combination

            jitter = 0.075
            random_jit = random.uniform(-jitter, jitter)

            x_jitter = [val + random_jit for val in x]

            scatt = group_bar_scatter.add_scatter(
                x=x_jitter,
                y=y,
                mode="markers",
                xaxis="x2",
                showlegend=False,
                marker={"color": "gray", "size": 4},
            )
        try:
            max_val = max(max(group1[i]), max(group2[i]))

        except:

            if len(group1[i]) == 0:
                max_val = max(group2[i])
            else:
                max_val = max(group1[i])

        sig_spacing = 0.3
        y_bar = (max_val + sig_spacing, max_val + sig_spacing)
        x_bar = np.mean(x)
        # add bar for sig
        group_bar_scatter.add_scatter(
            x=x,
            y=y_bar,
            mode="lines",
            showlegend=False,
            xaxis="x2",
            line={"color": "white", "width": 1.5},
        )

        group_bar_scatter.add_annotation(
            x=np.mean(x),
            y=y_bar[0] + (sig_spacing / 2),
            text=f"p={p_values[i]:.3}",
            showarrow=False,
            xref="x2",
        )

    group_bar_scatter.update_xaxes(showgrid=False)
    group_bar_scatter.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        yaxis_title=y_label,
        height=400,
        width=500,
        margin=dict(t=50, b=5, l=5, r=50, pad=5),
        bargap=bargap,
        legend=dict(x=1, y=0.1),
        yaxis_range=[-1, 1.5],
    )
    return group_bar_scatter


def correlation_matrix_subplot(
    data,
    x_label,
    y_label,
    title,
    plot_zones=None,
    zone_color=["lightblue", "blue"],
    zone_plots=[[8, 20], [21, 33]],
):

    if plot_zones is None:
        plot_zones = np.ones(len(data))

    assert len(zone_plots) == len(
        zone_color
    ), "number of zones does not match number of colors"

    spacing = 0.1
    # create plot for correlation matrixes
    corr_matrix_plt = make_subplots(
        rows=1, cols=len(data), shared_yaxes=True, horizontal_spacing=spacing
    )

    color_bar_x = [
        1 * (i + 1) / len(data) - (spacing / (i + 1)) for i in range(len(data))
    ]
    shapes = []

    for i in range(len(data)):
        corr_matrix_plt.add_trace(
            go.Heatmap(
                z=data[i],
                colorscale="Viridis",
                colorbar_x=color_bar_x[i],
                colorbar=dict(thickness=10),
            ),
            row=1,
            col=i + 1,
        )
        if plot_zones[i]:
            for j, zone in enumerate(zone_plots):
                shapes.append(
                    dict(
                        type="rect",
                        xref=f"x{i+1}",
                        yref=f"y{i+1}",
                        x0=zone[0],
                        y0=zone[0],
                        x1=zone[1],
                        y1=zone[1],
                        line_dash="dash",
                        line_color=zone_color[j],
                    )
                )

    corr_matrix_plt.update_layout(
        yaxis_title=y_label,
        xaxis_title=x_label,
        height=300,
        width=650,
        title_text=title,
        title_x=0.50,
        margin=dict(l=0, r=50, pad=5),
    )

    if any(plot_zones):
        corr_matrix_plt.update_layout(shapes=shapes)

    return corr_matrix_plt


def correlation_matrix_subplot_seaborn(
    data,
    x_label,
    plot_labels,
    title,
    fig_scale = 3.5,
    box_labels = ["Opto", 'Ctrl'],
    font_size = 12,
    plot_zones=None,
    zone_color=["lightblue", "blue"],
    colormap = 'viridis',
    zone_plots=[[8, 20], [21, 33]],
):

    if plot_zones is None:
        plot_zones = np.ones(len(data))

    assert len(zone_plots) == len(
        zone_color
    ), "number of zones does not match number of colors"

    num_plots = len(data)
    num_rows = 2
    num_cols = int(np.ceil(num_plots / 2))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_scale*num_cols, fig_scale*num_rows), sharey=True, sharex=True)

    # Flatten axes for easier iteration (in case of multiple rows)
    axes = axes.flatten()

    # Find global min and max for consistent color scaling
    vmin1 = min([np.min(d) for d in data[0:2]])
    vmax1 = max([np.max(d) for d in data[0:2]])

    vmin2 = min([np.min(d) for d in data[0:2]])
    vmax2 = max([np.max(d) for d in data[0:2]])

    vmins = [vmin2, vmin2, vmin2, vmin2]
    vmaxs = [vmax2, vmax2, vmax2, vmax2]

    for i, ax in enumerate(axes):
        sns.heatmap(
            data[i],
            ax=axes[i],
            cmap=colormap,
            vmin=vmins[i],  # Set the common color scale
            vmax=vmaxs[i],
            cbar=False,
            cbar_kws={"shrink": 0.5},
            square=True
        )
        axes[i].set_title(plot_labels[i], fontsize=font_size+3)
        axes[i].tick_params(length = 0, pad = 8)
        axes[i].set_xticks([0,40])
        axes[i].set_xticklabels([0, 110], rotation = 0, fontsize = font_size)
        
        axes[i].set_yticks([0,40])
        axes[i].set_yticklabels([0, 110], rotation = 0, fontsize = font_size)

        if i == 0 or i == 2:
            axes[i].set_ylabel(x_label, fontsize=font_size)
        
        if i == 2 or i == 3:
            axes[i].set_xlabel(x_label, fontsize=font_size)

        # Add plot zones (rectangles) to the heatmap
        if plot_zones[i]:
            for j, zone in enumerate(zone_plots):
                rect = Rectangle(
                    (zone[0], zone[0]),  # (x0, y0)
                    zone[1] - zone[0],   # width
                    zone[1] - zone[0],   # height
                    linewidth=2,
                    edgecolor=zone_color[j],
                    facecolor='none',
                    linestyle='--'
                )
                axes[i].add_patch(rect)
                # Adding the text "opto" for the first rectangle and "ctrl" for the second
                if j == 0:
                    axes[i].text(
                        zone[0]+3,  # Shift left for the text outside
                        zone[0] + zone[1] - zone[0] +2,  # Shift downward for the text outside
                        box_labels[j],
                        fontsize=font_size/2,
                        color=zone_color[j],
                        ha="center",
                        va="center",
                        #fontweight="bold"
                    )
                elif j == 1:
                    axes[i].text(
                        zone[0] + 2,  # Shift left for the text outside
                        zone[0] + zone[1] - zone[0] +2,  # Shift downward for the text outside
                        box_labels[j],
                        fontsize=font_size/2,
                        color=zone_color[j],
                        ha="center",
                        va="center",
                        #fontweight="bold"
                    )



    # Hide any unused axes if the number of plots isn't an exact multiple of 2
    for j in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Add a single color bar outside the subplots
    cbar_ax = fig.add_axes([0.90, 0.25, 0.03, 0.5])  # Position for the color bar
    norm = plt.Normalize(vmin=vmins[0], vmax=vmaxs[0])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Only needed for matplotlib < 3.1
    cbar = fig.colorbar(sm, cax=cbar_ax)

    cbar.set_ticks([vmins[0], vmaxs[0]])
    cbar.set_ticklabels([f'{vmins[0]:.2f}', f'{vmaxs[0]:.2f}'], fontsize = font_size-4)

    # Add title to the color bar along its longest length
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=0, fontsize=font_size)



    plt.suptitle(title, fontsize=font_size+4)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    #plt.subplots_adjust(top=0.85)  # Adjust to fit the title better
    return fig, axes



def plot_bar_stripplot_single(data, stats = None, y_title = "y title",
                              x_titles = ["Control \nZone", "Opto \nZone"], 
                              plot_type = 'boxplot', colormap = 'mako', 
                              point_size = 9, line_thickness = 4, 
                              group_spacing = 0.2, show_ns = True, fontsize_scale = 3,
                              plot_w = 3.75, plot_h = 7):



    new_palette = []
    palette_div = 3

    palette = sns.color_palette(colormap, palette_div)
    new_palette.append(palette[0])
    new_palette.append(palette[palette_div-1])
    new_palette_alpha = [t + (0.5,) for t in new_palette]

    sns.set_theme(style="ticks", font='Arial', font_scale=fontsize_scale)
    
    fig = plt.figure(figsize=[plot_w,plot_h])


    if plot_type == 'boxplot':
        # Create the box plot with error bars
       ax1 =  sns.boxplot(data=data, 
                    showfliers = False,
                    palette=new_palette,
                    linewidth = line_thickness/2,
                    width=0.7,
                    gap = group_spacing)
       
    elif plot_type == 'barplot':
        
        # Create the bar plot with error bars
       ax1= sns.barplot(data=data, 
                    palette=new_palette,
                    linewidth = line_thickness/2,
                    width=0.7,
                    errorbar='se', 
                    capsize=0.25, 
                    err_kws={'linewidth': line_thickness/2,'color': 'black'}, 
                    edgecolor = 'black',
                    gap = group_spacing
                    )

    # Change the border color of the bars
    new_palette_alpha = [color + (0.5,) for color in new_palette]
    for i, patch in enumerate(ax1.patches):
        patch.set_edgecolor('black')
        patch.set_linewidth(line_thickness/2)
        patch.set_facecolor(new_palette_alpha[i])

    df_melted = data.melt(var_name='Groups', value_name='Values')
    # add a swarm plot to the bars
    ax1 = sns.stripplot(x='Groups', 
                    y='Values', 
                    hue='Groups', 
                    palette=new_palette, 
                    data=df_melted,
                    edgecolor='black',
                    linewidth=0.5, 
                    jitter=0.3,
                    size=point_size, 
                    alpha=0.7)

    ax1.set(ylabel=y_title, xlabel = "")


    # add the significance values to the bar 
    max_val = np.max(data.iloc[:,0:2].max())*1.1
    plt.plot([0,1],[max_val+0.2,max_val+0.2],color='black', linewidth=line_thickness)
    plt.text(0.5, max_val+0.5, f'{stats}', va ='bottom', ha='center')


    ax1.spines['bottom'].set_linewidth(line_thickness)  # bottom axis
    ax1.spines['left'].set_linewidth(line_thickness)  # left axis
    ax1.tick_params(axis='y', which='major', width=line_thickness, length=10)
    ax1.tick_params(axis='x', width=0)
    ax1.set_xticks([0, 1], x_titles)
    ax1.set_ylim([0, max_val*1.25])
    ax1.set_xlim([-0.7, 1.7])


    sns.despine()
    # Adjust layout and spacing
    plt.tight_layout()
    return fig


def plot_ratemaps(rates_to_plot, sub_titles, title_for_data = None, opto_actual_bins = [8, 20]):

    # set some varibles for the plotting
    fontsize = 14
    number_bins = 40
    num_rates = rates_to_plot[0].shape[0]
    plot_opto = False
    figure_title = title_for_data
    subplot_titles = sub_titles

    subplot_dims = len(rates_to_plot)
    subplot_widths = np.append(np.ones(subplot_dims), 0.1)

    # create a subplot including one for the color bar 
    fig, axs = plt.subplots(1,subplot_dims+1, gridspec_kw={'width_ratios': subplot_widths}, figsize=(subplot_dims*3, 7))
    fig.suptitle(figure_title, fontsize=fontsize+2)

    for i, rates in enumerate(rates_to_plot):

        # plot the data as a seaborn heatmap
        im = sns.heatmap(np.array(rates), cmap='viridis', ax=axs[i], cbar=False)
        
        # change the x-axis settings
        axs[i].set_xticks(np.arange(0, number_bins+1, round(number_bins/2)))
        axs[i].set_xticklabels(np.arange(0, number_bins+1, round(number_bins/2)), rotation=0, fontsize = fontsize)
        
        # change the y-axis settings
        axs[i].set_title(subplot_titles[i], fontsize=fontsize+2)
        axs[i].set_ylim(0, num_rates)
        axs[i].tick_params(length = 0)
        axs[i].set_yticks([num_rates, 1])
        axs[i].set_yticklabels([1, num_rates], rotation=0, fontsize= fontsize)
        
        # set the titles on specific subplots
        axs[1].set_xlabel("Track position", fontsize = fontsize+2)
        axs[0].set_ylabel("Place cells", fontsize = fontsize+2)

        # Create the colorbar on the fourth axis
        cbar = fig.colorbar(im.collections[0], cax=axs[-1])
        cbar.ax.tick_params(labelsize=fontsize-1, length=0)
        cbar.set_label('max DF/F0', labelpad=-2, fontsize=fontsize-1)



    if plot_opto:
        for zone in opto_actual_bins:
            [axs[i].axvline(x = zone, linewidth=1.5, linestyle='--', color="tab:orange", alpha = 0.75) for i in range(len(rates_to_plot))]


    #plt.tight_layout()
    return fig




# def plot_paired_box(data, x, y, group, stats = None, connect_points=True, plot_type = 'boxplot', colormap = 'mako', 
#                        point_size = 9, line_thickness = 4, 
#                        group_spacing = 0.2, show_ns = True, fontsize_scale = 3, width_scale = 3.75, height = 7):


#     num_groups = len(data['Experiment'].unique())
#     new_palette = []
#     palette_div = 3

#     palette = sns.color_palette(colormap, palette_div)
#     new_palette.append(palette[0])
#     new_palette.append(palette[palette_div-1])
#     new_palette_alpha = [t + (0.5,) for t in new_palette]

#     sns.set_theme(style="ticks", font='Arial', font_scale=fontsize_scale)
    
#     fig = plt.figure(figsize=[width_scale*num_groups,height])

#     if plot_type == 'barplot':
#         # Create the bar plot with error bars
#         ax1 = sns.barplot(data=data,
#                         x = x,
#                         y = y,
#                         hue = group, 
#                         errorbar='se', 
#                         capsize=0.25, 
#                         err_kws={'linewidth': line_thickness/2,'color': 'black'}, 
#                         palette=new_palette_alpha,
#                         linewidth = line_thickness/2,
#                         #width=0.5,
#                         #alpha = 0.5,
#                         edgecolor = 'black',
#                         dodge = True,
#                         gap = group_spacing)
        
#     elif plot_type == 'boxplot':
        
#         ax1 = sns.boxplot(data=data,
#                         x = x,
#                         y = y,
#                         hue = group,  
#                         showfliers = False,
#                         palette=new_palette_alpha,
#                         linewidth = line_thickness/2,
#                         width=0.75,
#                         #alpha = 0.5,
#                         #edgecolor = 'black',
#                         dodge = True,
#                         gap = group_spacing)


#     new_palette_alpha = [t + (0.5,) for t in new_palette]
#     new_palette_alpha4 = [t + (0.5,) for t in new_palette for _ in range(num_groups)]
#     # # Change the border color of the bars
#     for i, patch in enumerate(ax1.patches):
#         if i < len(new_palette_alpha4):
#             patch.set_facecolor((new_palette_alpha4[i]))
    

#     if connect_points: 
#         # Display the individual data points on the plot with jitter
#         ax2 = sns.stripplot(data=data,
#                             x = x,
#                             y = y,
#                             hue = group, 
#                             palette=new_palette,  
#                             jitter=0,
#                             dodge = True, 
#                             edgecolor='black',
#                             linewidth=0.5, 
#                             size=point_size-1, 
#                             alpha=0.7,
#                             legend=False)
            
#         for name, group_data in data.groupby(x):
#             group_data_pivot = group_data.pivot(columns=group, values=y)
            
#             control_vals = group_data_pivot['Control-zone'].dropna().values
#             opto_vals = group_data_pivot['Opto-zone'].dropna().values

#             xval = list(data[x].unique()).index(name)
#             for i in range(len(control_vals)):
#                 plt.plot([xval-group_spacing, xval+group_spacing], [control_vals[i], opto_vals[i]], color='black', linewidth=line_thickness/3)

#     else:
#         # Display the individual data points on the plot with jitter
#         ax2 = sns.stripplot(data=data,color='gray', jitter=0.25, size=point_size, alpha=0.7)



#     labels = ax1.get_xticklabels()  
#     fontsize = labels[0].get_fontsize()

    
#     if stats is not None:
#         # Add significance to the graph 
#         max_val = np.max(data[y])
#         for i in range(num_groups):
#             pos = [i-group_spacing, i+group_spacing] 

#             if stats[i] == 'ns':
#                 if not show_ns:
#                     continue
#                 plt.text(i, max_val*1.25, f'{stats[i]}', ha='center', fontsize = fontsize*0.75)
#                 plt.plot(pos,[max_val*1.2,max_val*1.2],color='black', linewidth=line_thickness/2)
#             else:
#                 plt.text(i, max_val*1.17, f'{stats[i]}', ha='center')
#                 plt.plot(pos,[max_val*1.2,max_val*1.2],color='black', linewidth=line_thickness/2)

#     legend = ax2.get_legend()
#     for i, handle in enumerate(legend.legend_handles):
#         handle.set_facecolor(new_palette[i]+ (0.5,))

#     sns.move_legend(ax2, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, title=None, fontsize = fontsize*0.7)

#     sns.despine()

#     # Change the thickness of the axis lines

#     plt.gca().spines['top'].set_linewidth(line_thickness)  # top axis
#     plt.gca().spines['bottom'].set_linewidth(0)  # bottom axis
#     plt.gca().spines['left'].set_linewidth(line_thickness)  # left axis
#     plt.gca().spines['right'].set_linewidth(line_thickness)  # right axis
#     plt.tick_params(axis='y', which='major', width=line_thickness, length=10)
#     plt.tick_params(axis='x', which='major', width=0)
#     plt.tight_layout()
#     return fig



def plot_paired_box(data, x, y, group, pallette, point_pallette, stats = None, connect_points=True, plot_type = 'boxplot', colormap = 'mako', 
                       point_size = 9, line_thickness = 3, 
                       group_spacing = 0.2, show_ns = True, fontsize_scale = 3, 
                       width_scale = 3.75, height = 7, 
                       point_alpha = 0.5,
                       group_dv = 2, exp_groups = None):


    num_groups = len(data[x].unique())
    new_palette = []
    palette_div = 3
    sf = 0.2
    palette = sns.color_palette('mako', palette_div)
    new_palette.append(palette[0])
    new_palette.append(palette[palette_div-1])
    new_palette_alpha = [(x+sf, y+sf, z+sf) for x, y, z in new_palette]
    
    sns.set_theme(style="ticks", font='Arial', font_scale=fontsize_scale)
    
    if exp_groups is None: 
        exp_labels = list(data[x].unique())
        exp_groups = [exp_labels[:group_dv], exp_labels[group_dv:]]  # First two groups, and last two groups

    
        # Create the plot
    fig, ax1 = plt.subplots(figsize=[width_scale*num_groups,height])

    # Loop through the two groups and plot each with its respective palette
    for i, (exp_group, palette, pal_l) in enumerate(zip(exp_groups, pallette, point_pallette)):
        # Filter the data for the current group
        group_data = data[data[x].isin(exp_group)]



        sns.barplot(data=group_data,
                    x = x,
                    y = y,
                    hue = group, 
                    errorbar='se', 
                    capsize=0.25, 
                    err_kws={'linewidth': line_thickness/2,'color': 'black'}, 
                    palette=palette,
                    linewidth = line_thickness/2,
                    #width=0.7,
                    #alpha = 0.5,
                    edgecolor = 'black',
                    dodge = True,
                    gap = group_spacing,
                    saturation=1,
                    ax = ax1)
    
        
        sns.stripplot(data=group_data,
                x = x,
                y = y,
                hue = group, 
                palette=pal_l, 
                edgecolor='black',
                linewidth=0.5, 
                jitter=0,
                size=point_size-1, 
                alpha=point_alpha,
                dodge=True,
                legend = False, 
                ax=ax1)
        
        if connect_points:
            
            group_names = data[group].unique()

            for name, data_group in data.groupby(x):
                data_group_pivot = data_group.pivot(columns=group, values=y)
                
                control_vals = data_group_pivot[group_names[0]].dropna().values
                opto_vals = data_group_pivot[group_names[1]].dropna().values

                xval = list(data[x].unique()).index(name)
                for i in range(len(control_vals)):
                    plt.plot([xval-group_spacing, xval+group_spacing], [control_vals[i], opto_vals[i]], color='black', linewidth=line_thickness/3)


    labels = ax1.get_xticklabels()  
    fontsize = labels[0].get_fontsize()

    if stats is not None:
        # Add significance to the graph 
        max_val = np.max(data[y])
        for i in range(num_groups):
            pos = [i-group_spacing, i+group_spacing] 

            if stats[i] == 'ns':
                if not show_ns:
                    continue
                plt.text(i, max_val*1.25, f'{stats[i]}', ha='center', fontsize = fontsize)
                plt.plot(pos,[max_val*1.2,max_val*1.2],color='black', linewidth=line_thickness/2)
            else:
                plt.text(i, max_val*1.17, f'{stats[i]}', ha='center',fontsize = fontsize*1.75)
                plt.plot(pos,[max_val*1.2,max_val*1.2],color='black', linewidth=line_thickness/2)

    # legend = ax2.get_legend()
    # for i, handle in enumerate(legend.legend_handles):
    #     handle.set_facecolor(new_palette[i]+ (0.5,))

    sns.move_legend(ax1, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, title=None, fontsize = fontsize*0.7)

    sns.despine()
    #plt.ylim([-0.1,1])
    #plt.ylabel("")
    # Change the thickness of the axis lines

    plt.gca().spines['top'].set_linewidth(line_thickness)  # top axis
    plt.gca().spines['bottom'].set_linewidth(0)  # bottom axis
    plt.gca().spines['left'].set_linewidth(line_thickness)  # left axis
    plt.gca().spines['right'].set_linewidth(line_thickness)  # right axis
    plt.tick_params(axis='y', which='major', width=line_thickness, length=10)
    plt.tick_params(axis='x', which='major', width=0)
    #plt.tight_layout()
    return fig



def plot_bar_stripplot(data, x, y, group, pallette, point_pallette, stats = None, 
                       plot_type = 'boxplot',
                       point_size = 9, line_thickness = 3, 
                       group_spacing = 0.2, show_ns = True, 
                       fontsize_scale = 3, point_alpha = 0.5,
                       width_scale = 3.75, height = 7, plot_xorigin = False,
                       notch = True, exp_groups = None

                       ):


    sf = -0.2
    num_groups = len(data['Experiment'].unique())
    
    
    if exp_groups is None:
        exp_labels = list(data[x].unique())
        exp_groups = [exp_labels[:2], exp_labels[2:]]  # First two groups, and last two groups

    sns.set_theme(style="ticks", font='Arial', font_scale=fontsize_scale)
    


    # Create the plot
    fig, ax1 = plt.subplots(figsize=[width_scale*num_groups,height])

    # Loop through the two groups and plot each with its respective palette
    for i, (exp_group, palette, pal_l) in enumerate(zip(exp_groups, pallette, point_pallette)):
        # Filter the data for the current group
        group_data = data[data[x].isin(exp_group)]
        
        if plot_type == 'boxplot':
            # Plot the current group
            sns.boxplot(
                data=group_data,
                x=x,
                y=y,
                hue=group,
                showfliers=False,
                palette=palette,
                linewidth=line_thickness / 2,
                dodge=True,
                notch=notch,
                gap=group_spacing,
                saturation = 1,
                ax=ax1)  # Use the same axes for both plots
        elif plot_type == 'barplot': 
            sns.barplot(data=group_data,
                x = x,
                y = y,
                hue = group, 
                errorbar='se', 
                capsize=0.25, 
                err_kws={'linewidth': line_thickness/2,'color': 'black'}, 
                palette=palette,
                linewidth = line_thickness/2,
                edgecolor = 'black',
                dodge = True,
                gap = group_spacing,
                saturation = 1,
                ax = ax1)
        else:
            sns.violinplot(data=group_data, 
                           x=x, 
                           y=y, 
                           hue=group, 
                           split=True, 
                           palette = palette,
                           gap=group_spacing,
                           linewidth = line_thickness/2,
                           edgecolor = 'black', 
                           inner="quart",
                           ax = ax1
                           )

        
        sns.stripplot(data=group_data,
                x = x,
                y = y,
                hue = group, 
                palette=pal_l, 
                edgecolor='black',
                linewidth=0.5, 
                jitter=0.25,
                size=point_size-1, 
                alpha=point_alpha,
                dodge=True,
                legend = False,
                ax=ax1)
        


    labels = ax1.get_xticklabels()  
    fontsize = labels[0].get_fontsize()

    
    if stats is not None:
        # Add significance to the graph 
        max_val = np.max(data[y])
        for i in range(num_groups):
            pos = [i-0.2, i+0.2] 

            if stats[i] == 'ns':
                if not show_ns:
                    continue
                plt.text(i, max_val*1.25, f'{stats[i]}', ha='center', fontsize = fontsize)
                plt.plot(pos,[max_val*1.2,max_val*1.2],color='black', linewidth=line_thickness/2)
            else:
                plt.text(i, max_val*1.17, f'{stats[i]}', ha='center', fontsize = fontsize*1.75)
                plt.plot(pos,[max_val*1.2,max_val*1.2],color='black', linewidth=line_thickness/2)

    # add horizontal line at zero
    if plot_xorigin:
        ax1.axhline(y=0, color='black', linewidth=line_thickness/2, linestyle='--',)
    ax1.set_xlabel('')

    sns.move_legend(ax1, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False, title=None, fontsize = fontsize*0.7)

    sns.despine()

    # Change the thickness of the axis lines

    plt.gca().spines['top'].set_linewidth(line_thickness)  # top axis
    plt.gca().spines['bottom'].set_linewidth(0)  # bottom axis
    plt.gca().spines['left'].set_linewidth(line_thickness)  # left axis
    plt.gca().spines['right'].set_linewidth(line_thickness)  # right axis
    plt.tick_params(axis='y', which='major', width=line_thickness, length=10)
    plt.tick_params(axis='x', which='major', width=0)
    #plt.tight_layout()
    return fig



