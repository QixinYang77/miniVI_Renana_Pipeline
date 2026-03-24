import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter, label, center_of_mass
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def calculate_spatial_information(firing_rate_map, occupancy_map):
    """Calculates spatial information in bits per second."""
    epsilon = 1e-15
    occupancy_prob = occupancy_map / (np.sum(occupancy_map) + epsilon)
    mean_firing_rate = np.sum(firing_rate_map * occupancy_prob)
    
    if mean_firing_rate <= epsilon:
        return 0.0
        
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_ratio = firing_rate_map / mean_firing_rate
        log_term = np.log2(rate_ratio + epsilon)
        
    info_per_bin = firing_rate_map * log_term * occupancy_prob
    spatial_info = np.sum(info_per_bin[np.isfinite(info_per_bin)])
    return spatial_info

def add_scale_bar(ax, length=10, label="10 cm", vertical_pad=2, side_pad=2):
    """Adds a scale bar below the bottom-right corner of the axis."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_start = xlim[1] - length - side_pad
    x_end = xlim[1] - side_pad
    y_pos = ylim[0] - vertical_pad
    ax.plot([x_start, x_end], [y_pos, y_pos], color='black', linewidth=2, clip_on=False)
    ax.text((x_start + x_end) / 2, y_pos - (vertical_pad * 0.5), label, 
            ha='center', va='top', fontsize=6, color='black', clip_on=False)


def analyze_place_cell_subset(x_full, y_full, spikes_list, subset_indices, 
                              bins, frame_rate, smooth_sigma=2.0, num_shuffles=1000, 
                              place_field_threshold=0.2, min_field_bins=5):
    """
    Analyzes a specific subset of time indices.
    Saves a binary 'place_field_mask' where:
      1. Rate > threshold * peak
      2. Field size >= min_field_bins (connected components)
    """
    
    # 1. Extract Trajectory for this Subset
    x_sub = x_full[subset_indices]
    y_sub = y_full[subset_indices]
    
    # 2. Calculate Occupancy for this Subset
    occ_counts, _, _ = np.histogram2d(x_sub, y_sub, bins=bins)
    occ_map = occ_counts / frame_rate # in seconds
    #occ_map_raw = occ_map.copy()

    # Smooth occupancy slightly to avoid division artifacts
    occ_map = gaussian_filter(occ_map, sigma=2, mode='constant')
    #occ_map[occ_map<=0.1]=0
    
    results = []
    total_frames = len(x_full)
    
    print(f"Analyzing subset ({len(subset_indices)} frames)...")

    for i, cell_spikes in enumerate(spikes_list):
        # --- A. Create Boolean Spike Train ---
        binary_train_full = np.zeros(total_frames, dtype=bool)
        binary_train_full[cell_spikes] = True
        binary_train_sub = binary_train_full[subset_indices]
        
        # --- B. Calculate Rate Map ---
        spikes_in_sub_indices = np.where(binary_train_sub)[0]
        x_spk = x_sub[spikes_in_sub_indices]
        y_spk = y_sub[spikes_in_sub_indices]
        
        spike_map, _, _ = np.histogram2d(x_spk, y_spk, bins=bins)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            raw_map = spike_map / occ_map
            raw_map[np.isnan(raw_map)] = 0
            raw_map[np.isinf(raw_map)] = 0
            
        # Smooth Map
        smooth_map = gaussian_filter(raw_map, sigma=smooth_sigma, mode='constant')
        smooth_map[occ_map == 0] = np.nan # Mask unvisited
        
        # --- C. Define Place Field (Binary Mask with Size Filter) ---
        peak_rate = np.nanmax(smooth_map)
        place_field_mask = np.zeros_like(smooth_map, dtype=bool)
        
        if not (np.isnan(peak_rate) or peak_rate == 0):
            # 1. Thresholding
            threshold_val = place_field_threshold * peak_rate
            with np.errstate(invalid='ignore'):
                raw_field_mask = smooth_map > threshold_val
            
            # 2. Filter by Size (Connected Components)
            labeled_array, num_features = label(raw_field_mask)
            for feature_idx in range(1, num_features + 1):
                # Count bins in this feature
                if np.sum(labeled_array == feature_idx) >= min_field_bins:
                    place_field_mask[labeled_array == feature_idx] = True
        
        # --- D. Spatial Information ---
        si = calculate_spatial_information(raw_map, occ_map)
        
        # --- E. Shuffling ---
        shuffled_si_values = []
        if len(spikes_in_sub_indices) > 0:
            for _ in range(num_shuffles):
                shift = np.random.randint(1, len(binary_train_sub))
                shuf_bool = np.roll(binary_train_sub, shift=shift)
                
                shuf_x = x_sub[shuf_bool]
                shuf_y = y_sub[shuf_bool]
                
                shuf_spk_map, _, _ = np.histogram2d(shuf_x, shuf_y, bins=bins)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    shuf_rate = shuf_spk_map / occ_map
                    shuf_rate[np.isnan(shuf_rate)] = 0
                
                shuffled_si_values.append(calculate_spatial_information(shuf_rate, occ_map))
        else:
            shuffled_si_values = [0] * num_shuffles
            
        shuffled_si_arr = np.array(shuffled_si_values)
        p_val = np.sum(shuffled_si_arr >= si) / num_shuffles
        
        results.append({
            'cell_id': i,
            'spikes_x': x_spk,
            'spikes_y': y_spk,
            'rate_map': smooth_map,
            'place_field_mask': place_field_mask,
            'peak_rate': peak_rate,
            'si': si,
            'p_value': p_val,
            'shuffle_dist': shuffled_si_arr
        })

    return {
        'results': results,
        'x_traj': x_sub,
        'y_traj': y_sub,
        'occupancy': occ_map
    }

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================

def plot_comparison_results(data_run, data_slow, extent, save_folder=None):
    """
    Plots 2x3 Grid.
    Draws enclosed contours for significant Place Fields.
    """
    if save_folder:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
        
    x_run, y_run = data_run['x_traj'], data_run['y_traj']
    x_slow, y_slow = data_slow['x_traj'], data_slow['y_traj']

    # --- Helper to draw enclosed boundaries ---
    def draw_enclosed_boundary(ax, mask, extent, color='magenta'):
        """
        Pads the mask to ensure contours close, draws contours for each place field,
        and adds a text label indicating the field number at its center.
        """
        # 1. Label connected components (distinct place fields)
        labeled_mask, num_features = label(mask)
        
        # Calculate bin sizes for coordinate conversion
        nx, ny = mask.shape
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        bin_x = x_range / nx
        bin_y = y_range / ny
        
        # 2. Iterate over each detected field to draw and label
        for i in range(1, num_features + 1):
            # Create a boolean mask for just this specific field
            field_mask = (labeled_mask == i)
            
            # --- A. Draw Contour (with padding) ---
            # Pad with 1 pixel border of False
            padded_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
            padded_mask[1:-1, 1:-1] = field_mask
            
            # Adjust extent to match the padded array
            # extent is (x0, x1, y0, y1)
            padded_extent = (extent[0] - bin_x, extent[1] + bin_x,
                            extent[2] - bin_y, extent[3] + bin_y)
            
            # Draw contour
            # Note: mask.T is used because we usually imshow(mask.T) with origin='lower'
            ax.contour(padded_mask.T, levels=[0.5], colors=color, 
                    linewidths=1, extent=padded_extent, origin='lower')
            
            # --- B. Add Label at Center ---
            # Calculate center of mass in array indices (x_idx, y_idx)
            com = center_of_mass(field_mask)
            
            if com:
                center_x_idx, center_y_idx = com
                
                # Convert indices to data coordinates (cm)
                # Position = Start + (Index * Bin_Size) + (Half_Bin_Size to center)
                center_x = extent[0] + (center_x_idx * bin_x) + (bin_x / 2)
                center_y = extent[2] + (center_y_idx * bin_y) + (bin_y / 2)
                
                # Add the text label
                ax.text(center_x, center_y, str(i), color=color, fontsize=6, 
                        ha='center', va='center', fontweight='bold', clip_on=True)
    
    for i in range(len(data_run['results'])):
        res_run = data_run['results'][i]
        res_slow = data_slow['results'][i]
        cell_id = res_run['cell_id']
        
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), gridspec_kw={'width_ratios': [1, 1, 0.4]})
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # --- ROW 1: RUNNING ---
        cmap = plt.get_cmap('jet').copy()
        cmap.set_bad(color='white')
        
        # A. Rate Map
        map_run = res_run['rate_map']
        masked_map_run = ma.masked_where(np.isnan(map_run), map_run)
        im1 = axes[0,0].imshow(masked_map_run.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        
        # Draw ENCLOSED Contour (Running)
        if res_run['p_value'] < 0.05 and np.any(res_run['place_field_mask']):
            draw_enclosed_boundary(axes[0,0], res_run['place_field_mask'], extent)

        axes[0,0].set_title(f"Cell {cell_id} (Run >=2)\nPeak: {res_run['peak_rate']:.1f} Hz", fontsize=7)
        
        # B. Trajectory
        axes[0,1].plot(x_run, y_run, color='#bbbbbb', alpha=0.5, linewidth=0.3, label='Run Path', zorder=1)
        axes[0,1].scatter(res_run['spikes_x'], res_run['spikes_y'], s=1.5, c='red', alpha=0.5, linewidths=0, zorder=2)
        
        # C. SI
        axes[0,2].hist(res_run['shuffle_dist'], bins=30, color='gray', alpha=0.6, density=True, linewidth=0)
        axes[0,2].axvline(res_run['si'], color='red', linestyle='--', linewidth=1.0)
        axes[0,2].text(0.95, 0.95, f"p={res_run['p_value']:.3f}\nSI={res_run['si']:.2f}", 
                       transform=axes[0,2].transAxes, ha='right', va='top', fontsize=6, color='red')
        axes[0,2].set_ylabel("Prob")

        # --- ROW 2: STATIONARY ---
        # A. Rate Map
        map_slow = res_slow['rate_map']
        masked_map_slow = ma.masked_where(np.isnan(map_slow), map_slow)
        im2 = axes[1,0].imshow(masked_map_slow.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        
        # Draw ENCLOSED Contour (Slow)
        if res_slow['p_value'] < 0.05 and np.any(res_slow['place_field_mask']):
            draw_enclosed_boundary(axes[1,0], res_slow['place_field_mask'], extent)
            
        axes[1,0].set_title(f"(Slow <2)\nPeak: {res_slow['peak_rate']:.1f} Hz", fontsize=7)
        
        # B. Trajectory
        axes[1,1].plot(x_slow, y_slow, color='cyan', alpha=0.3, linewidth=0.3, label='Slow Path', zorder=1)
        axes[1,1].scatter(res_slow['spikes_x'], res_slow['spikes_y'], s=1.5, c='red', alpha=0.5, linewidths=0, zorder=2)
        
        # C. SI Comparison
        axes[1,2].hist(res_slow['shuffle_dist'], bins=30, color='#dddddd', alpha=0.6, density=True, linewidth=0)
        axes[1,2].axvline(res_slow['si'], color='blue', linestyle='--', linewidth=1.0)
        axes[1,2].text(0.95, 0.95, f"p={res_slow['p_value']:.3f}\nSI={res_slow['si']:.2f}", 
                       transform=axes[1,2].transAxes, ha='right', va='top', fontsize=6, color='blue')
        axes[1,2].set_xlabel("SI (bits/sec)")
        axes[1,2].set_ylabel("Prob")

        # --- FORMATTING ---
        for ax, im in zip([axes[0,0], axes[1,0]], [im1, im2]):
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_linewidth(0.5)

        for r in [0,1]:
            for c in [0,1]:
                ax = axes[r,c]
                ax.set_aspect('equal', adjustable='box')
                ax.set_anchor('C')
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xticks([])
                ax.set_yticks([])
                add_scale_bar(ax, length=10)
                
            div = make_axes_locatable(axes[r,1])
            cax_d = div.append_axes("right", size="5%", pad=0.05)
            cax_d.axis('off')
            
            axes[r,2].spines['top'].set_visible(False)
            axes[r,2].spines['right'].set_visible(False)

        plt.tight_layout()
        
        if save_folder:
            prefix = "PlaceCell" if res_run['p_value'] < 0.05 else "NonPlace"
            fname = f"{prefix}_{cell_id}.pdf"
            plt.savefig(os.path.join(save_folder, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def get_epoch_indices(boolean_mask, min_duration_frames):
    """
    Returns indices where the boolean_mask is True for at least min_duration_frames consecutively.
    """
    # Label connected components (contiguous True regions)
    labeled_array, num_features = label(boolean_mask)
    
    valid_indices = []
    
    # Iterate through each detected epoch
    for i in range(1, num_features + 1):
        # Get indices of the current epoch
        epoch_indices = np.where(labeled_array == i)[0]
        
        # Check duration
        if len(epoch_indices) >= min_duration_frames:
            valid_indices.extend(epoch_indices)
            
    return np.sort(np.array(valid_indices, dtype=int))

def plot_combined_spike_types(data_run_all, data_run_CS, data_run_SS,
                               data_slow_all, data_slow_CS, data_slow_SS,
                               extent, save_folder=None, spike_labels=None):
    """
    Combines place cell analysis figures for 3 spike types (all_spikes, all_CS_spikes, refined_SS)
    and 2 behavioral states (walking/running vs resting/slow).
    
    Layout: 6 rows x 3 columns
      - Rows 0-2: Walking (all_spikes, all_CS_spikes, refined_SS)
      - Rows 3-5: Resting (all_spikes, all_CS_spikes, refined_SS)
      - Columns: Rate Map, Trajectory+Spikes, SI Histogram
    
    Parameters
    ----------
    data_run_all, data_run_CS, data_run_SS : dict
        Output from analyze_place_cell_subset for walking condition
    data_slow_all, data_slow_CS, data_slow_SS : dict
        Output from analyze_place_cell_subset for resting condition
    extent : tuple
        (x_min, x_max, y_min, y_max) for plotting
    save_folder : str, optional
        Folder to save PDFs. If None, displays plots.
    spike_labels : list, optional
        Labels for spike types. Default: ['All Spikes', 'CS Spikes', 'SS Spikes']
    """
    if save_folder:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    
    if spike_labels is None:
        spike_labels = ['All Spikes', 'CS Spikes', 'SS Spikes']
    
    # Helper function to draw enclosed boundaries
    def draw_enclosed_boundary(ax, mask, extent, color='magenta'):
        labeled_mask, num_features = label(mask)
        nx, ny = mask.shape
        x_range = extent[1] - extent[0]
        y_range = extent[3] - extent[2]
        bin_x = x_range / nx
        bin_y = y_range / ny
        
        for i in range(1, num_features + 1):
            field_mask = (labeled_mask == i)
            padded_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
            padded_mask[1:-1, 1:-1] = field_mask
            padded_extent = (extent[0] - bin_x, extent[1] + bin_x,
                            extent[2] - bin_y, extent[3] + bin_y)
            ax.contour(padded_mask.T, levels=[0.5], colors=color, 
                      linewidths=1, extent=padded_extent, origin='lower')
            com = center_of_mass(field_mask)
            if com:
                center_x_idx, center_y_idx = com
                center_x = extent[0] + (center_x_idx * bin_x) + (bin_x / 2)
                center_y = extent[2] + (center_y_idx * bin_y) + (bin_y / 2)
                ax.text(center_x, center_y, str(i), color=color, fontsize=5, 
                        ha='center', va='center', fontweight='bold', clip_on=True)
    
    # Organize data: [(run_data, slow_data), ...]
    run_data_list = [data_run_all, data_run_CS, data_run_SS]
    slow_data_list = [data_slow_all, data_slow_CS, data_slow_SS]
    
    num_cells = len(data_run_all['results'])
    
    for cell_idx in range(num_cells):
        cell_id = data_run_all['results'][cell_idx]['cell_id']
        
        # Create 6x3 figure
        fig, axes = plt.subplots(6, 3, figsize=(7.2, 12), 
                                 gridspec_kw={'width_ratios': [1, 1, 0.4], 
                                              'hspace': 0.35, 'wspace': 0.25})
        
        cmap = plt.get_cmap('jet').copy()
        cmap.set_bad(color='white')
        
        # --- TOP 3 ROWS: WALKING ---
        for row_idx, (run_data, label_str) in enumerate(zip(run_data_list, spike_labels)):
            res = run_data['results'][cell_idx]
            x_traj = run_data['x_traj']
            y_traj = run_data['y_traj']
            
            # A. Rate Map
            rate_map = res['rate_map']
            masked_map = ma.masked_where(np.isnan(rate_map), rate_map)
            im = axes[row_idx, 0].imshow(masked_map.T, origin='lower', extent=extent, 
                                         cmap=cmap, interpolation='nearest')
            
            if res['p_value'] < 0.05 and np.any(res['place_field_mask']):
                draw_enclosed_boundary(axes[row_idx, 0], res['place_field_mask'], extent)
            
            if row_idx == 0:
                axes[row_idx, 0].set_title(f"Cell {cell_id} - Walking\n{label_str}\nPeak: {res['peak_rate']:.1f} Hz", fontsize=7)
            else:
                axes[row_idx, 0].set_title(f"{label_str}\nPeak: {res['peak_rate']:.1f} Hz", fontsize=7)
            
            # Add colorbar
            div = make_axes_locatable(axes[row_idx, 0])
            cax = div.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=5)
            cb.outline.set_linewidth(0.5)
            
            # B. Trajectory + Spikes
            axes[row_idx, 1].plot(x_traj, y_traj, color='#bbbbbb', alpha=0.5, linewidth=0.3, zorder=1)
            axes[row_idx, 1].scatter(res['spikes_x'], res['spikes_y'], s=1.5, c='red', 
                                     alpha=0.5, linewidths=0, zorder=2)
            
            div = make_axes_locatable(axes[row_idx, 1])
            cax_d = div.append_axes("right", size="5%", pad=0.05)
            cax_d.axis('off')
            
            # C. SI Histogram
            axes[row_idx, 2].hist(res['shuffle_dist'], bins=30, color='gray', alpha=0.6, 
                                  density=True, linewidth=0)
            axes[row_idx, 2].axvline(res['si'], color='red', linestyle='--', linewidth=1.0)
            axes[row_idx, 2].text(0.95, 0.95, f"p={res['p_value']:.3f}\nSI={res['si']:.2f}", 
                                  transform=axes[row_idx, 2].transAxes, ha='right', va='top', 
                                  fontsize=5, color='red')
            axes[row_idx, 2].set_ylabel("Prob", fontsize=6)
            axes[row_idx, 2].spines['top'].set_visible(False)
            axes[row_idx, 2].spines['right'].set_visible(False)
            axes[row_idx, 2].tick_params(labelsize=5)
        
        # --- BOTTOM 3 ROWS: RESTING ---
        for row_offset, (slow_data, label_str) in enumerate(zip(slow_data_list, spike_labels)):
            row_idx = row_offset + 3
            res = slow_data['results'][cell_idx]
            x_traj = slow_data['x_traj']
            y_traj = slow_data['y_traj']
            
            # A. Rate Map
            rate_map = res['rate_map']
            masked_map = ma.masked_where(np.isnan(rate_map), rate_map)
            im = axes[row_idx, 0].imshow(masked_map.T, origin='lower', extent=extent, 
                                         cmap=cmap, interpolation='nearest')
            
            if res['p_value'] < 0.05 and np.any(res['place_field_mask']):
                draw_enclosed_boundary(axes[row_idx, 0], res['place_field_mask'], extent)
            
            if row_offset == 0:
                axes[row_idx, 0].set_title(f"Resting\n{label_str}\nPeak: {res['peak_rate']:.1f} Hz", fontsize=7)
            else:
                axes[row_idx, 0].set_title(f"{label_str}\nPeak: {res['peak_rate']:.1f} Hz", fontsize=7)
            
            # Add colorbar
            div = make_axes_locatable(axes[row_idx, 0])
            cax = div.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=5)
            cb.outline.set_linewidth(0.5)
            
            # B. Trajectory + Spikes
            axes[row_idx, 1].plot(x_traj, y_traj, color='cyan', alpha=0.3, linewidth=0.3, zorder=1)
            axes[row_idx, 1].scatter(res['spikes_x'], res['spikes_y'], s=1.5, c='red', 
                                     alpha=0.5, linewidths=0, zorder=2)
            
            div = make_axes_locatable(axes[row_idx, 1])
            cax_d = div.append_axes("right", size="5%", pad=0.05)
            cax_d.axis('off')
            
            # C. SI Histogram
            axes[row_idx, 2].hist(res['shuffle_dist'], bins=30, color='#dddddd', alpha=0.6, 
                                  density=True, linewidth=0)
            axes[row_idx, 2].axvline(res['si'], color='blue', linestyle='--', linewidth=1.0)
            axes[row_idx, 2].text(0.95, 0.95, f"p={res['p_value']:.3f}\nSI={res['si']:.2f}", 
                                  transform=axes[row_idx, 2].transAxes, ha='right', va='top', 
                                  fontsize=5, color='blue')
            axes[row_idx, 2].set_ylabel("Prob", fontsize=6)
            axes[row_idx, 2].set_xlabel("SI (bits/sec)", fontsize=6)
            axes[row_idx, 2].spines['top'].set_visible(False)
            axes[row_idx, 2].spines['right'].set_visible(False)
            axes[row_idx, 2].tick_params(labelsize=5)
        
        # --- Formatting for all rate map and trajectory axes ---
        for row_idx in range(6):
            for col_idx in [0, 1]:
                ax = axes[row_idx, col_idx]
                ax.set_aspect('equal', adjustable='box')
                ax.set_anchor('C')
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                ax.set_xticks([])
                ax.set_yticks([])
                add_scale_bar(ax, length=10, vertical_pad=1.5, side_pad=1.5)
        
        plt.tight_layout()
        
        if save_folder:
            # Check if any condition is significant for the first spike type (all_spikes)
            res_run_all = data_run_all['results'][cell_idx]
            prefix = "PlaceCell" if res_run_all['p_value'] < 0.05 else "NonPlace"
            fname = f"{prefix}_{cell_id}_combined.pdf"
            plt.savefig(os.path.join(save_folder, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def batch_combined_place_cell_analysis(x_neural, y_neural, all_spikes, all_CS_spikes, refined_SS,
                                        speed, traces, frame_rate, figure_folder,
                                        width_real=35.5, height_real=20, bin_size=1.5,
                                        place_field_threshold=0.4, min_field_bins=5,
                                        spike_labels=None):
    """
    Runs place cell analysis for 3 spike types and combines results into single figures.
    
    Creates 6-row figures per cell:
      - Top 3 rows: Walking condition (all_spikes, all_CS_spikes, refined_SS)
      - Bottom 3 rows: Resting condition (all_spikes, all_CS_spikes, refined_SS)
    
    Parameters
    ----------
    x_neural, y_neural : ndarray
        Position coordinates
    all_spikes, all_CS_spikes, refined_SS : list of ndarray
        Spike indices for each spike type (list of arrays, one per cell)
    speed : ndarray
        Speed at each frame
    traces : ndarray
        Neural traces (n_cells x n_frames)
    frame_rate : float
        Frame rate in Hz
    figure_folder : str
        Output folder for combined PDFs
    width_real, height_real : float
        Arena dimensions in cm
    bin_size : float
        Spatial bin size in cm
    place_field_threshold : float
        Threshold for place field detection (fraction of peak rate)
    min_field_bins : int
        Minimum number of bins for a place field
    spike_labels : list, optional
        Labels for the 3 spike types
    """
    arena_size = (width_real, height_real)
    bins = [np.arange(0, arena_size[0] + bin_size, bin_size),
            np.arange(0, arena_size[1] + bin_size, bin_size)]
    extent = (0, width_real, 0, height_real)
    
    # Define valid frames
    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed)) & (~np.isnan(traces).any(axis=0))
    
    n_total = len(x_neural)
    n_valid = np.sum(valid_frames)
    n_nan = n_total - n_valid
    
    print(f"Total frames: {n_total}")
    print(f"Valid frames (no NaN in x/y/speed): {n_valid}")
    print(f"Frames excluded due to NaN: {n_nan} ({100.0 * n_nan / n_total:.2f}%)")
    
    # Define running and slow indices
    idx_run = np.where((speed >= 2) & valid_frames)[0]
    idx_slow = np.where((speed < 2) & valid_frames)[0]
    
    print(f"Walking frames (speed>=2, NaN-free): {len(idx_run)}")
    print(f"Resting frames (speed<2, NaN-free): {len(idx_slow)}")
    
    # Run analysis for all 3 spike types
    print("\n--- Analyzing All Spikes (Walking) ---")
    results_run_all = analyze_place_cell_subset(
        x_neural, y_neural, all_spikes, subset_indices=idx_run,
        bins=bins, frame_rate=frame_rate,
        place_field_threshold=place_field_threshold, min_field_bins=min_field_bins
    )
    
    print("\n--- Analyzing All Spikes (Resting) ---")
    results_slow_all = analyze_place_cell_subset(
        x_neural, y_neural, all_spikes, subset_indices=idx_slow,
        bins=bins, frame_rate=frame_rate,
        place_field_threshold=place_field_threshold, min_field_bins=min_field_bins
    )
    
    print("\n--- Analyzing CS Spikes (Walking) ---")
    results_run_CS = analyze_place_cell_subset(
        x_neural, y_neural, all_CS_spikes, subset_indices=idx_run,
        bins=bins, frame_rate=frame_rate,
        place_field_threshold=place_field_threshold, min_field_bins=min_field_bins
    )
    
    print("\n--- Analyzing CS Spikes (Resting) ---")
    results_slow_CS = analyze_place_cell_subset(
        x_neural, y_neural, all_CS_spikes, subset_indices=idx_slow,
        bins=bins, frame_rate=frame_rate,
        place_field_threshold=place_field_threshold, min_field_bins=min_field_bins
    )
    
    print("\n--- Analyzing SS Spikes (Walking) ---")
    results_run_SS = analyze_place_cell_subset(
        x_neural, y_neural, refined_SS, subset_indices=idx_run,
        bins=bins, frame_rate=frame_rate,
        place_field_threshold=place_field_threshold, min_field_bins=min_field_bins
    )
    
    print("\n--- Analyzing SS Spikes (Resting) ---")
    results_slow_SS = analyze_place_cell_subset(
        x_neural, y_neural, refined_SS, subset_indices=idx_slow,
        bins=bins, frame_rate=frame_rate,
        place_field_threshold=place_field_threshold, min_field_bins=min_field_bins
    )
    
    # Plot combined results
    print("\n--- Generating Combined Figures ---")
    plot_combined_spike_types(
        results_run_all, results_run_CS, results_run_SS,
        results_slow_all, results_slow_CS, results_slow_SS,
        extent, save_folder=figure_folder, spike_labels=spike_labels
    )
    print(f"Combined plots saved to {figure_folder}")


def batch_place_cell_analysis(x_neural, y_neural, spikes, speed, traces, frame_rate, figure_folder, 
                              width_real = 35.5, height_real = 20, bin_size = 1.5, 
                              speed_threshold = 2, place_field_threshold = 0.4, min_field_bins = 5):

    arena_size = (width_real, height_real)
    #arena_size = (np.ceil(np.max(x_neural)), np.ceil(np.max(y_neural)))
    bins = [np.arange(0, arena_size[0] + bin_size, bin_size),
            np.arange(0, arena_size[1] + bin_size, bin_size)]
    extent = (0, width_real, 0, height_real)

    # 2. Define Indices
    # ==== 1. GLOBAL NaN HANDLING ====
    # Any frame where x_neural, y_neural or speed is NaN will be excluded from *all* analyses
    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed)) & (~np.isnan(traces).any(axis=0))

    n_total = len(x_neural)
    n_valid = np.sum(valid_frames)
    n_nan = n_total - n_valid

    print(f"Total frames: {n_total}")
    print(f"Valid frames (no NaN in x/y/speed): {n_valid}")
    print(f"Frames excluded due to NaN: {n_nan} ({100.0 * n_nan / n_total:.2f}%)")

    # ---- Simple version: no minimum epoch length ----
    idx_run = np.where((speed >= speed_threshold) & valid_frames)[0]
    idx_slow = np.where((speed < speed_threshold) & valid_frames)[0]

    print(f"Running frames (speed>={speed_threshold}, NaN-free): {len(idx_run)}")
    print(f"Stationary frames (speed<{speed_threshold}, NaN-free): {len(idx_slow)}")

    
    # 3. RUN 1: VALID / RUNNING
    print("Running Analysis for Speed >= 2...")
    results_run = analyze_place_cell_subset(
        x_neural, y_neural, spikes, 
        subset_indices=idx_run, 
        bins=bins, frame_rate=frame_rate, 
        place_field_threshold=place_field_threshold,
        min_field_bins=min_field_bins
    )

    # 4. RUN 2: SLOW / STATIONARY
    print("Running Analysis for Speed < 2...")
    results_slow = analyze_place_cell_subset(
        x_neural, y_neural, spikes, 
        subset_indices=idx_slow, 
        bins=bins, frame_rate=frame_rate, 
        place_field_threshold=place_field_threshold,
        min_field_bins=min_field_bins
    )

    # 5. Plot Comparison
    plot_comparison_results(results_run, results_slow, extent, save_folder=figure_folder)
    print(f"Plots saved to {figure_folder}")

    return results_run, results_slow

# Head direction analysis 

def analyze_head_direction(hd_angles, spikes_list, subset_indices, frame_rate, 
                           bin_size_deg=15, num_shuffles=1000):
    """
    Analyze head direction tuning for each cell.
    
    Parameters:
    -----------
    hd_angles : array
        Head direction angles in degrees for all frames
    spikes_list : list
        List of spike indices for each cell
    subset_indices : array
        Indices of frames to include in analysis
    frame_rate : float
        Frame rate in Hz
    bin_size_deg : float
        Bin size in degrees (default 15)
    num_shuffles : int
        Number of shuffles for significance testing
        
    Returns:
    --------
    dict with results for each cell
    """
    
    # Define bins (0 to 360 degrees)
    bin_edges = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)
    
    # Extract HD angles for this subset
    hd_sub = hd_angles[subset_indices]
    
    # Calculate occupancy (time spent in each HD bin)
    occ_counts, _ = np.histogram(hd_sub, bins=bin_edges)
    occ_time = occ_counts / frame_rate  # in seconds
    
    total_frames = len(hd_angles)
    results = []
    
    print(f"Analyzing head direction ({len(subset_indices)} frames)...")
    
    for i, cell_spikes in enumerate(tqdm(spikes_list)):
        # Create binary spike train
        binary_train_full = np.zeros(total_frames, dtype=bool)
        binary_train_full[cell_spikes] = True
        binary_train_sub = binary_train_full[subset_indices]
        
        # Get HD angles at spike times
        spike_indices_in_sub = np.where(binary_train_sub)[0]
        hd_at_spikes = hd_sub[spike_indices_in_sub]
        
        # Calculate spike counts per HD bin
        spike_counts, _ = np.histogram(hd_at_spikes, bins=bin_edges)
        
        # Calculate firing rate per bin
        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = spike_counts / occ_time
            firing_rate[np.isnan(firing_rate)] = 0
            firing_rate[np.isinf(firing_rate)] = 0
        
        # Calculate mean vector length (MVL) - measure of tuning strength
        # Convert bin centers to radians
        theta_rad = np.deg2rad(bin_centers)
        
        # Weight by firing rate
        total_rate = np.sum(firing_rate)
        if total_rate > 0:
            rate_normalized = firing_rate / total_rate
            mean_x = np.sum(rate_normalized * np.cos(theta_rad))
            mean_y = np.sum(rate_normalized * np.sin(theta_rad))
            mvl = np.sqrt(mean_x**2 + mean_y**2)
            preferred_direction = np.rad2deg(np.arctan2(mean_y, mean_x)) % 360
        else:
            mvl = 0
            preferred_direction = np.nan
        
        # Shuffle test for MVL
        shuffled_mvl = []
        if len(spike_indices_in_sub) > 0:
            for _ in range(num_shuffles):
                shift = np.random.randint(1, len(binary_train_sub))
                shuf_train = np.roll(binary_train_sub, shift=shift)
                
                shuf_hd = hd_sub[shuf_train]
                shuf_counts, _ = np.histogram(shuf_hd, bins=bin_edges)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    shuf_rate = shuf_counts / occ_time
                    shuf_rate[np.isnan(shuf_rate)] = 0
                    shuf_rate[np.isinf(shuf_rate)] = 0
                
                shuf_total = np.sum(shuf_rate)
                if shuf_total > 0:
                    shuf_norm = shuf_rate / shuf_total
                    shuf_x = np.sum(shuf_norm * np.cos(theta_rad))
                    shuf_y = np.sum(shuf_norm * np.sin(theta_rad))
                    shuffled_mvl.append(np.sqrt(shuf_x**2 + shuf_y**2))
                else:
                    shuffled_mvl.append(0)
        else:
            shuffled_mvl = [0] * num_shuffles
        
        shuffled_mvl = np.array(shuffled_mvl)
        p_value = np.sum(shuffled_mvl >= mvl) / num_shuffles
        
        results.append({
            'cell_id': i,
            'firing_rate': firing_rate,
            'bin_centers': bin_centers,
            'bin_edges': bin_edges,
            'mvl': mvl,
            'preferred_direction': preferred_direction,
            'peak_rate': np.max(firing_rate),
            'mean_rate': np.mean(firing_rate),
            'p_value': p_value,
            'shuffle_dist': shuffled_mvl,
            'occupancy': occ_time
        })
    
    return {
        'results': results,
        'bin_centers': bin_centers,
        'bin_edges': bin_edges,
        'occupancy': occ_time
    }


def plot_hd_comparison(data_run, data_slow, save_folder=None):
    """
    Plot head direction tuning curves as polar plots.
    Compares running vs stationary epochs.
    """
    if save_folder:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    
    bin_centers_rad = np.deg2rad(data_run['bin_centers'])
    
    for i in range(len(data_run['results'])):
        res_run = data_run['results'][i]
        res_slow = data_slow['results'][i]
        cell_id = res_run['cell_id']
        
        fig, axes = plt.subplots(2, 3, figsize=(9, 6), 
                                  subplot_kw={'projection': 'polar'},
                                  gridspec_kw={'width_ratios': [1, 1, 0.8]})
        
        # Convert 3rd column to regular axes for histograms
        for row in range(2):
            axes[row, 2].remove()
        ax_hist_run = fig.add_subplot(2, 3, 3)
        ax_hist_slow = fig.add_subplot(2, 3, 6)
        
        # --- ROW 1: RUNNING ---
        ax_polar_run = axes[0, 0]
        ax_polar_run2 = axes[0, 1]
        
        # Close the polar plot by appending first value
        fr_run = np.append(res_run['firing_rate'], res_run['firing_rate'][0])
        theta_closed = np.append(bin_centers_rad, bin_centers_rad[0])
        
        # Polar plot - firing rate
        ax_polar_run.plot(theta_closed, fr_run, 'b-', linewidth=1.5)
        ax_polar_run.fill(theta_closed, fr_run, alpha=0.3, color='blue')
        ax_polar_run.set_theta_zero_location('N')
        ax_polar_run.set_theta_direction(-1)
        ax_polar_run.set_title(f'Cell {cell_id} (Run ≥2 cm/s)\nPeak: {res_run["peak_rate"]:.1f} Hz', fontsize=8, pad=10)
        
        # Draw preferred direction arrow if significant
        if res_run['p_value'] < 0.05 and not np.isnan(res_run['preferred_direction']):
            pref_rad = np.deg2rad(res_run['preferred_direction'])
            ax_polar_run.annotate('', xy=(pref_rad, res_run['peak_rate']*0.8), 
                                   xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        # Second polar plot - occupancy normalized
        occ_run = np.append(data_run['occupancy'], data_run['occupancy'][0])
        ax_polar_run2.plot(theta_closed, occ_run, 'gray', linewidth=1, alpha=0.7)
        ax_polar_run2.fill(theta_closed, occ_run, alpha=0.2, color='gray')
        ax_polar_run2.set_theta_zero_location('N')
        ax_polar_run2.set_theta_direction(-1)
        ax_polar_run2.set_title(f'Occupancy (s)', fontsize=8, pad=10)
        
        # Histogram - MVL shuffle distribution
        ax_hist_run.hist(res_run['shuffle_dist'], bins=30, color='gray', alpha=0.6, density=True)
        ax_hist_run.axvline(res_run['mvl'], color='red', linestyle='--', linewidth=1.5)
        ax_hist_run.set_xlabel('MVL', fontsize=8)
        ax_hist_run.set_ylabel('Density', fontsize=8)
        ax_hist_run.text(0.95, 0.95, f"MVL={res_run['mvl']:.3f}\np={res_run['p_value']:.3f}\nPref={res_run['preferred_direction']:.0f}°", 
                         transform=ax_hist_run.transAxes, ha='right', va='top', fontsize=7,
                         color='red' if res_run['p_value'] < 0.05 else 'black')
        ax_hist_run.spines['top'].set_visible(False)
        ax_hist_run.spines['right'].set_visible(False)
        
        # --- ROW 2: STATIONARY ---
        ax_polar_slow = axes[1, 0]
        ax_polar_slow2 = axes[1, 1]
        
        fr_slow = np.append(res_slow['firing_rate'], res_slow['firing_rate'][0])
        
        ax_polar_slow.plot(theta_closed, fr_slow, 'g-', linewidth=1.5)
        ax_polar_slow.fill(theta_closed, fr_slow, alpha=0.3, color='green')
        ax_polar_slow.set_theta_zero_location('N')
        ax_polar_slow.set_theta_direction(-1)
        ax_polar_slow.set_title(f'(Slow <2 cm/s)\nPeak: {res_slow["peak_rate"]:.1f} Hz', fontsize=8, pad=10)
        
        if res_slow['p_value'] < 0.05 and not np.isnan(res_slow['preferred_direction']):
            pref_rad = np.deg2rad(res_slow['preferred_direction'])
            ax_polar_slow.annotate('', xy=(pref_rad, res_slow['peak_rate']*0.8), 
                                    xytext=(0, 0),
                                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        occ_slow = np.append(data_slow['occupancy'], data_slow['occupancy'][0])
        ax_polar_slow2.plot(theta_closed, occ_slow, 'gray', linewidth=1, alpha=0.7)
        ax_polar_slow2.fill(theta_closed, occ_slow, alpha=0.2, color='gray')
        ax_polar_slow2.set_theta_zero_location('N')
        ax_polar_slow2.set_theta_direction(-1)
        ax_polar_slow2.set_title(f'Occupancy (s)', fontsize=8, pad=10)
        
        ax_hist_slow.hist(res_slow['shuffle_dist'], bins=30, color='gray', alpha=0.6, density=True)
        ax_hist_slow.axvline(res_slow['mvl'], color='blue', linestyle='--', linewidth=1.5)
        ax_hist_slow.set_xlabel('MVL', fontsize=8)
        ax_hist_slow.set_ylabel('Density', fontsize=8)
        ax_hist_slow.text(0.95, 0.95, f"MVL={res_slow['mvl']:.3f}\np={res_slow['p_value']:.3f}\nPref={res_slow['preferred_direction']:.0f}°", 
                          transform=ax_hist_slow.transAxes, ha='right', va='top', fontsize=7,
                          color='blue' if res_slow['p_value'] < 0.05 else 'black')
        ax_hist_slow.spines['top'].set_visible(False)
        ax_hist_slow.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_folder:
            # Prefix based on significance
            if res_run['p_value'] < 0.05:
                prefix = "HD_tuned"
            else:
                prefix = "HD_nontuned"
            fname = f"{prefix}_cell{cell_id}.pdf"
            plt.savefig(os.path.join(save_folder, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# Combined Place Field + Head Direction Analysis

def get_in_place_field_indices(x, y, place_field_mask, bins):
    """
    Determine which position samples fall inside the place field.
    
    Parameters:
    -----------
    x, y : arrays
        Position coordinates
    place_field_mask : 2D bool array
        Binary mask of place field (shape matches histogram bins)
    bins : list of arrays
        Bin edges for x and y
        
    Returns:
    --------
    Boolean array indicating which samples are inside the place field
    """
    # Digitize positions to find which bin each sample falls into
    x_bin_idx = np.digitize(x, bins[0]) - 1  # -1 because digitize returns 1-indexed
    y_bin_idx = np.digitize(y, bins[1]) - 1
    
    # Clip to valid range
    x_bin_idx = np.clip(x_bin_idx, 0, place_field_mask.shape[0] - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, place_field_mask.shape[1] - 1)
    
    # Check if each sample is inside place field
    in_field = place_field_mask[x_bin_idx, y_bin_idx]
    
    return in_field


def analyze_hd_simple(hd_angles, spikes_list, subset_indices, frame_rate, 
                      bin_size_deg=15, num_shuffles=500, smooth_sigma=1, occ_threshold=1.0):
    """
    HD analysis with shuffle test for significance.
    Returns tuning curve, MVL, p-value, and HD occupancy probability.
    
    Parameters:
    -----------
    smooth_sigma : float
        Gaussian smoothing sigma in bins (default 1 bin)
    occ_threshold : float
        Minimum occupancy time (in seconds) for a bin to be included.
        Bins with less occupancy will have firing rate set to 0.
    """
    bin_edges = np.arange(0, 360 + bin_size_deg, bin_size_deg)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_bins = len(bin_centers)
    
    # Helper function for circular smoothing
    def circular_smooth(data, sigma):
        """Apply Gaussian smoothing with circular boundary conditions."""
        if sigma <= 0:
            return data
        # Pad data circularly (wrap around)
        pad_size = int(3 * sigma) + 1
        padded = np.concatenate([data[-pad_size:], data, data[:pad_size]])
        # Apply Gaussian filter
        smoothed_padded = gaussian_filter(padded.astype(float), sigma=sigma, mode='nearest')
        # Extract the original portion
        return smoothed_padded[pad_size:-pad_size]
    
    if len(subset_indices) == 0:
        # Return empty results if no indices
        results = []
        for i, cell_spikes in enumerate(spikes_list):
            results.append({
                'cell_id': i,
                'firing_rate': np.zeros(len(bin_centers)),
                'bin_centers': bin_centers,
                'mvl': 0,
                'preferred_direction': np.nan,
                'peak_rate': 0,
                'n_spikes': 0,
                'hd_occupancy_prob': np.zeros(len(bin_centers)),
                'p_value': 1.0
            })
        return results
    
    hd_sub = hd_angles[subset_indices]
    occ_counts, _ = np.histogram(hd_sub, bins=bin_edges)
    occ_time = occ_counts / frame_rate
    
    # Create occupancy mask for bins with sufficient sampling
    occ_valid = occ_time >= occ_threshold
    
    # Calculate HD occupancy probability (for visualization) - also smoothed
    total_occ = np.sum(occ_counts)
    if total_occ > 0:
        hd_occupancy_prob = occ_counts / total_occ
        hd_occupancy_prob = circular_smooth(hd_occupancy_prob, smooth_sigma)
    else:
        hd_occupancy_prob = np.zeros(len(bin_centers))
    
    total_frames = len(hd_angles)
    theta_rad = np.deg2rad(bin_centers)
    results = []
    
    for i, cell_spikes in enumerate(spikes_list):
        binary_train_full = np.zeros(total_frames, dtype=bool)
        binary_train_full[cell_spikes] = True
        binary_train_sub = binary_train_full[subset_indices]
        
        spike_indices_in_sub = np.where(binary_train_sub)[0]
        hd_at_spikes = hd_sub[spike_indices_in_sub]
        
        spike_counts, _ = np.histogram(hd_at_spikes, bins=bin_edges)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = spike_counts / occ_time
            firing_rate[np.isnan(firing_rate)] = 0
            firing_rate[np.isinf(firing_rate)] = 0
        
        # Apply occupancy threshold - set firing rate to 0 for under-sampled bins
        firing_rate[~occ_valid] = 0
        
        # Apply circular smoothing to firing rate
        firing_rate = circular_smooth(firing_rate, smooth_sigma)
        
        total_rate = np.sum(firing_rate)
        
        if total_rate > 0:
            rate_normalized = firing_rate / total_rate
            mean_x = np.sum(rate_normalized * np.cos(theta_rad))
            mean_y = np.sum(rate_normalized * np.sin(theta_rad))
            mvl = np.sqrt(mean_x**2 + mean_y**2)
            preferred_direction = np.rad2deg(np.arctan2(mean_y, mean_x)) % 360
        else:
            mvl = 0
            preferred_direction = np.nan
        
        # Shuffle test for MVL significance
        shuffled_mvl = []
        if len(spike_indices_in_sub) > 0:
            for _ in range(num_shuffles):
                shift = np.random.randint(1, len(binary_train_sub))
                shuf_train = np.roll(binary_train_sub, shift=shift)
                
                shuf_hd = hd_sub[shuf_train]
                shuf_counts, _ = np.histogram(shuf_hd, bins=bin_edges)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    shuf_rate = shuf_counts / occ_time
                    shuf_rate[np.isnan(shuf_rate)] = 0
                    shuf_rate[np.isinf(shuf_rate)] = 0
                
                # Apply occupancy threshold to shuffled data
                shuf_rate[~occ_valid] = 0
                
                # Apply same smoothing to shuffled data
                shuf_rate = circular_smooth(shuf_rate, smooth_sigma)
                
                shuf_total = np.sum(shuf_rate)
                if shuf_total > 0:
                    shuf_norm = shuf_rate / shuf_total
                    shuf_x = np.sum(shuf_norm * np.cos(theta_rad))
                    shuf_y = np.sum(shuf_norm * np.sin(theta_rad))
                    shuffled_mvl.append(np.sqrt(shuf_x**2 + shuf_y**2))
                else:
                    shuffled_mvl.append(0)
            p_value = np.sum(np.array(shuffled_mvl) >= mvl) / num_shuffles
        else:
            p_value = 1.0
        
        results.append({
            'cell_id': i,
            'firing_rate': firing_rate,
            'bin_centers': bin_centers,
            'mvl': mvl,
            'preferred_direction': preferred_direction,
            'peak_rate': np.max(firing_rate),
            'n_spikes': len(hd_at_spikes),
            'hd_occupancy_prob': hd_occupancy_prob,
            'p_value': p_value
        })
    
    return results


def add_circular_colorbar(fig, ax, cmap_name='hsv', size=0.08):
    """
    Add a circular colorbar (color wheel) for head direction.
    """
    # Get axis position
    pos = ax.get_position()
    
    # Create a small axes for the color wheel
    cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + pos.height*0.3, size, size], projection='polar')
    
    # Create color wheel
    n_segments = 360
    theta = np.linspace(0, 2*np.pi, n_segments + 1)
    
    # Get colors from colormap
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / n_segments) for i in range(n_segments)]
    
    # Plot colored segments
    for i in range(n_segments):
        cax.fill_between([theta[i], theta[i+1]], 0, 1, color=colors[i])
    
    # Configure the polar axis
    cax.set_theta_zero_location('N')
    cax.set_theta_direction(-1)
    cax.set_yticks([])
    cax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    cax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=5)
    cax.tick_params(pad=1)
    cax.set_title('HD', fontsize=6, pad=2)
    
    return cax


def plot_combined_pf_hd(results_run, results_slow, hd_results_run, hd_results_slow,
                        x_neural, y_neural, hd_angles_neural, spikes, 
                        idx_run, idx_slow, bins, extent, frame_rate,
                        save_folder=None):
    """
    Create combined PDF for each cell with:
    - Row 0: Place field rate maps (Running | Stationary)
    - Row 1: Trajectory + spikes colored by HD (Running | Stationary)
    - Row 2: HD In Place Field (Running | Stationary)
    - Row 3: HD Outside Place Field (Running | Stationary)
    - Row 4: HD All Positions (Running | Stationary)
    """
    if save_folder:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    
    bin_centers_rad = np.deg2rad(hd_results_run[0]['bin_centers'])
    n_cells = len(results_run['results'])
    
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='white')
    
    for cell_idx in range(n_cells):
        res_run = results_run['results'][cell_idx]
        res_slow = results_slow['results'][cell_idx]
        cell_id = res_run['cell_id']
        
        # Check if cell has significant place field in either condition
        has_pf_run = res_run['p_value'] < 0.05 and np.any(res_run['place_field_mask'])
        has_pf_slow = res_slow['p_value'] < 0.05 and np.any(res_slow['place_field_mask'])
        has_any_pf = has_pf_run or has_pf_slow
        
        # Determine figure layout - always 5 rows for consistent layout
        fig = plt.figure(figsize=(8, 16))
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 0.9, 0.9, 0.9], hspace=0.5, wspace=0.3)
        
        # =====================================================================
        # ROW 0: PLACE FIELD RATE MAPS
        # =====================================================================
        
        # Running - Rate Map
        ax_pf_run = fig.add_subplot(gs[0, 0])
        map_run = res_run['rate_map']
        masked_map_run = ma.masked_where(np.isnan(map_run), map_run)
        im1 = ax_pf_run.imshow(masked_map_run.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        ax_pf_run.set_title(f"Cell {cell_id} - Running (≥2 cm/s)\nPeak: {res_run['peak_rate']:.1f} Hz, SI: {res_run['si']:.2f}, p={res_run['p_value']:.3f}", fontsize=8)
        ax_pf_run.set_aspect('equal')
        ax_pf_run.set_xticks([])
        ax_pf_run.set_yticks([])
        
        div1 = make_axes_locatable(ax_pf_run)
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        cb1 = plt.colorbar(im1, cax=cax1)
        cb1.ax.tick_params(labelsize=6)
        
        # Draw place field contour if significant
        if has_pf_run:
            labeled_mask, num_features = label(res_run['place_field_mask'])
            nx, ny = res_run['place_field_mask'].shape
            bin_x = (extent[1] - extent[0]) / nx
            bin_y = (extent[3] - extent[2]) / ny
            
            for f_idx in range(1, num_features + 1):
                field_mask = (labeled_mask == f_idx)
                padded_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
                padded_mask[1:-1, 1:-1] = field_mask
                padded_extent = (extent[0] - bin_x, extent[1] + bin_x, extent[2] - bin_y, extent[3] + bin_y)
                ax_pf_run.contour(padded_mask.T, levels=[0.5], colors='magenta', linewidths=1.5, extent=padded_extent, origin='lower')
        
        # Stationary - Rate Map
        ax_pf_slow = fig.add_subplot(gs[0, 1])
        map_slow = res_slow['rate_map']
        masked_map_slow = ma.masked_where(np.isnan(map_slow), map_slow)
        im2 = ax_pf_slow.imshow(masked_map_slow.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        ax_pf_slow.set_title(f"Cell {cell_id} - Stationary (<2 cm/s)\nPeak: {res_slow['peak_rate']:.1f} Hz, SI: {res_slow['si']:.2f}, p={res_slow['p_value']:.3f}", fontsize=8)
        ax_pf_slow.set_aspect('equal')
        ax_pf_slow.set_xticks([])
        ax_pf_slow.set_yticks([])
        
        div2 = make_axes_locatable(ax_pf_slow)
        cax2 = div2.append_axes("right", size="5%", pad=0.05)
        cb2 = plt.colorbar(im2, cax=cax2)
        cb2.ax.tick_params(labelsize=6)
        
        if has_pf_slow:
            labeled_mask, num_features = label(res_slow['place_field_mask'])
            nx, ny = res_slow['place_field_mask'].shape
            bin_x = (extent[1] - extent[0]) / nx
            bin_y = (extent[3] - extent[2]) / ny
            
            for f_idx in range(1, num_features + 1):
                field_mask = (labeled_mask == f_idx)
                padded_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
                padded_mask[1:-1, 1:-1] = field_mask
                padded_extent = (extent[0] - bin_x, extent[1] + bin_x, extent[2] - bin_y, extent[3] + bin_y)
                ax_pf_slow.contour(padded_mask.T, levels=[0.5], colors='magenta', linewidths=1.5, extent=padded_extent, origin='lower')
        
        # =====================================================================
        # ROW 1: TRAJECTORY + SPIKES (colored by head direction)
        # =====================================================================
        
        # Get spike indices and corresponding HD angles for this cell
        cell_spike_indices = spikes[cell_idx]
        total_frames = len(hd_angles_neural)
        
        # Running - get spike HD angles
        binary_train_full_run = np.zeros(total_frames, dtype=bool)
        binary_train_full_run[cell_spike_indices] = True
        binary_train_run = binary_train_full_run[idx_run]
        spike_indices_run = np.where(binary_train_run)[0]
        hd_at_spikes_run = hd_angles_neural[idx_run][spike_indices_run]
        
        # Normalize HD to 0-1 for colormap
        hd_colors_run = hd_at_spikes_run / 360.0
        
        # Running - Trajectory with spikes colored by HD
        ax_traj_run = fig.add_subplot(gs[1, 0])
        ax_traj_run.plot(results_run['x_traj'], results_run['y_traj'], color='#cccccc', alpha=0.5, linewidth=0.3)
        if len(res_run['spikes_x']) > 0:
            ax_traj_run.scatter(res_run['spikes_x'], res_run['spikes_y'], s=3, c=hd_colors_run, 
                               cmap='hsv', alpha=0.8, linewidths=0, vmin=0, vmax=1)
        ax_traj_run.set_xlim(extent[0], extent[1])
        ax_traj_run.set_ylim(extent[2], extent[3])
        ax_traj_run.set_aspect('equal')
        ax_traj_run.set_xticks([])
        ax_traj_run.set_yticks([])
        ax_traj_run.set_title("Trajectory + Spikes by HD (Running)", fontsize=8)
        
        # Add circular colorbar for running
        add_circular_colorbar(fig, ax_traj_run)
        
        # Stationary - get spike HD angles
        binary_train_slow = binary_train_full_run[idx_slow]
        spike_indices_slow = np.where(binary_train_slow)[0]
        hd_at_spikes_slow = hd_angles_neural[idx_slow][spike_indices_slow]
        
        # Normalize HD to 0-1 for colormap
        hd_colors_slow = hd_at_spikes_slow / 360.0
        
        # Stationary - Trajectory with spikes colored by HD
        ax_traj_slow = fig.add_subplot(gs[1, 1])
        ax_traj_slow.plot(results_slow['x_traj'], results_slow['y_traj'], color='#cccccc', alpha=0.5, linewidth=0.3)
        if len(res_slow['spikes_x']) > 0:
            ax_traj_slow.scatter(res_slow['spikes_x'], res_slow['spikes_y'], s=3, c=hd_colors_slow, 
                                cmap='hsv', alpha=0.8, linewidths=0, vmin=0, vmax=1)
        ax_traj_slow.set_xlim(extent[0], extent[1])
        ax_traj_slow.set_ylim(extent[2], extent[3])
        ax_traj_slow.set_aspect('equal')
        ax_traj_slow.set_xticks([])
        ax_traj_slow.set_yticks([])
        ax_traj_slow.set_title("Trajectory + Spikes by HD (Stationary)", fontsize=8)
        
        # Add circular colorbar for stationary
        add_circular_colorbar(fig, ax_traj_slow)
        
        # =====================================================================
        # HD ANALYSIS - Separate by place field and behavioral state
        # Each behavioral state uses its OWN place field mask
        # =====================================================================
        
        theta_closed = np.append(bin_centers_rad, bin_centers_rad[0])
        
        # Helper function to add scaled occupancy probability overlay
        def add_occupancy_overlay(ax, occ_prob, theta_closed, peak_rate):
            """Add gray occupancy probability overlay scaled to firing rate axis."""
            occ_closed = np.append(occ_prob, occ_prob[0])
            if peak_rate > 0:
                occ_scaled = occ_closed * (peak_rate * 0.5) / np.max(occ_closed) if np.max(occ_closed) > 0 else occ_closed
            else:
                occ_scaled = occ_closed
            ax.fill(theta_closed, occ_scaled, alpha=0.15, color='gray', zorder=0)
            ax.plot(theta_closed, occ_scaled, color='gray', linewidth=0.8, alpha=0.5, zorder=0)
        
        # Helper function for significance stars
        def get_sig_stars(p_value):
            """Return significance stars based on p-value."""
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ''
        
        # === RUNNING HD Analysis (Left Column) ===
        if has_pf_run:
            # Use RUNNING place field mask for running HD analysis
            x_run = x_neural[idx_run]
            y_run = y_neural[idx_run]
            in_field_run = get_in_place_field_indices(x_run, y_run, res_run['place_field_mask'], bins)
            idx_run_in = idx_run[in_field_run]
            idx_run_out = idx_run[~in_field_run]
            
            # Compute HD tuning for running in/out of running PF
            hd_run_in = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_run_in, frame_rate)[0]
            hd_run_out = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_run_out, frame_rate)[0]
            
            # ROW 2 Left: Running In Place Field
            ax_hd_run_in = fig.add_subplot(gs[2, 0], projection='polar')
            add_occupancy_overlay(ax_hd_run_in, hd_run_in['hd_occupancy_prob'], theta_closed, hd_run_in['peak_rate'])
            fr_run_in = np.append(hd_run_in['firing_rate'], hd_run_in['firing_rate'][0])
            ax_hd_run_in.plot(theta_closed, fr_run_in, 'b-', linewidth=1.5, zorder=2)
            ax_hd_run_in.fill(theta_closed, fr_run_in, alpha=0.3, color='blue', zorder=1)
            ax_hd_run_in.set_theta_zero_location('N')
            ax_hd_run_in.set_theta_direction(-1)
            sig_run_in = get_sig_stars(hd_run_in['p_value'])
            ax_hd_run_in.set_title(f"HD In PF_Run (Running){sig_run_in}\nPeak: {hd_run_in['peak_rate']:.1f} Hz, MVL: {hd_run_in['mvl']:.3f}, p={hd_run_in['p_value']:.3f}", fontsize=8, pad=10)
            
            # ROW 3 Left: Running Outside Place Field
            ax_hd_run_out = fig.add_subplot(gs[3, 0], projection='polar')
            add_occupancy_overlay(ax_hd_run_out, hd_run_out['hd_occupancy_prob'], theta_closed, hd_run_out['peak_rate'])
            fr_run_out = np.append(hd_run_out['firing_rate'], hd_run_out['firing_rate'][0])
            ax_hd_run_out.plot(theta_closed, fr_run_out, 'purple', linewidth=1.5, zorder=2)
            ax_hd_run_out.fill(theta_closed, fr_run_out, alpha=0.3, color='purple', zorder=1)
            ax_hd_run_out.set_theta_zero_location('N')
            ax_hd_run_out.set_theta_direction(-1)
            sig_run_out = get_sig_stars(hd_run_out['p_value'])
            ax_hd_run_out.set_title(f"HD Outside PF_Run (Running){sig_run_out}\nPeak: {hd_run_out['peak_rate']:.1f} Hz, MVL: {hd_run_out['mvl']:.3f}, p={hd_run_out['p_value']:.3f}", fontsize=8, pad=10)
        
        # === STATIONARY HD Analysis (Right Column) ===
        if has_pf_slow:
            # Use STATIONARY place field mask for stationary HD analysis
            x_slow = x_neural[idx_slow]
            y_slow = y_neural[idx_slow]
            in_field_slow = get_in_place_field_indices(x_slow, y_slow, res_slow['place_field_mask'], bins)
            idx_slow_in = idx_slow[in_field_slow]
            idx_slow_out = idx_slow[~in_field_slow]
            
            # Compute HD tuning for stationary in/out of stationary PF
            hd_slow_in = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_slow_in, frame_rate)[0]
            hd_slow_out = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_slow_out, frame_rate)[0]
            
            # ROW 2 Right: Stationary In Place Field
            ax_hd_slow_in = fig.add_subplot(gs[2, 1], projection='polar')
            add_occupancy_overlay(ax_hd_slow_in, hd_slow_in['hd_occupancy_prob'], theta_closed, hd_slow_in['peak_rate'])
            fr_slow_in = np.append(hd_slow_in['firing_rate'], hd_slow_in['firing_rate'][0])
            ax_hd_slow_in.plot(theta_closed, fr_slow_in, 'g-', linewidth=1.5, zorder=2)
            ax_hd_slow_in.fill(theta_closed, fr_slow_in, alpha=0.3, color='green', zorder=1)
            ax_hd_slow_in.set_theta_zero_location('N')
            ax_hd_slow_in.set_theta_direction(-1)
            sig_slow_in = get_sig_stars(hd_slow_in['p_value'])
            ax_hd_slow_in.set_title(f"HD In PF_Slow (Stationary){sig_slow_in}\nPeak: {hd_slow_in['peak_rate']:.1f} Hz, MVL: {hd_slow_in['mvl']:.3f}, p={hd_slow_in['p_value']:.3f}", fontsize=8, pad=10)
            
            # ROW 3 Right: Stationary Outside Place Field
            ax_hd_slow_out = fig.add_subplot(gs[3, 1], projection='polar')
            add_occupancy_overlay(ax_hd_slow_out, hd_slow_out['hd_occupancy_prob'], theta_closed, hd_slow_out['peak_rate'])
            fr_slow_out = np.append(hd_slow_out['firing_rate'], hd_slow_out['firing_rate'][0])
            ax_hd_slow_out.plot(theta_closed, fr_slow_out, 'orange', linewidth=1.5, zorder=2)
            ax_hd_slow_out.fill(theta_closed, fr_slow_out, alpha=0.3, color='orange', zorder=1)
            ax_hd_slow_out.set_theta_zero_location('N')
            ax_hd_slow_out.set_theta_direction(-1)
            sig_slow_out = get_sig_stars(hd_slow_out['p_value'])
            ax_hd_slow_out.set_title(f"HD Outside PF_Slow (Stationary){sig_slow_out}\nPeak: {hd_slow_out['peak_rate']:.1f} Hz, MVL: {hd_slow_out['mvl']:.3f}, p={hd_slow_out['p_value']:.3f}", fontsize=8, pad=10)
        
        # === ROW 4: HD All Positions (always plotted, irrespective of place field) ===
        hd_run_cell = hd_results_run[cell_idx]
        hd_slow_cell = hd_results_slow[cell_idx]
        
        # Left: Running All Positions
        ax_hd_run_all = fig.add_subplot(gs[4, 0], projection='polar')
        add_occupancy_overlay(ax_hd_run_all, hd_run_cell['occupancy'] / np.sum(hd_run_cell['occupancy']) if np.sum(hd_run_cell['occupancy']) > 0 else hd_run_cell['occupancy'], theta_closed, hd_run_cell['peak_rate'])
        fr_run_all = np.append(hd_run_cell['firing_rate'], hd_run_cell['firing_rate'][0])
        ax_hd_run_all.plot(theta_closed, fr_run_all, 'darkblue', linewidth=1.5, zorder=2)
        ax_hd_run_all.fill(theta_closed, fr_run_all, alpha=0.3, color='darkblue', zorder=1)
        ax_hd_run_all.set_theta_zero_location('N')
        ax_hd_run_all.set_theta_direction(-1)
        sig_run_all = get_sig_stars(hd_run_cell['p_value'])
        ax_hd_run_all.set_title(f"HD All Positions (Running){sig_run_all}\nPeak: {hd_run_cell['peak_rate']:.1f} Hz, MVL: {hd_run_cell['mvl']:.3f}, p={hd_run_cell['p_value']:.3f}", fontsize=8, pad=10)
        
        # Right: Stationary All Positions
        ax_hd_slow_all = fig.add_subplot(gs[4, 1], projection='polar')
        add_occupancy_overlay(ax_hd_slow_all, hd_slow_cell['occupancy'] / np.sum(hd_slow_cell['occupancy']) if np.sum(hd_slow_cell['occupancy']) > 0 else hd_slow_cell['occupancy'], theta_closed, hd_slow_cell['peak_rate'])
        fr_slow_all = np.append(hd_slow_cell['firing_rate'], hd_slow_cell['firing_rate'][0])
        ax_hd_slow_all.plot(theta_closed, fr_slow_all, 'darkgreen', linewidth=1.5, zorder=2)
        ax_hd_slow_all.fill(theta_closed, fr_slow_all, alpha=0.3, color='darkgreen', zorder=1)
        ax_hd_slow_all.set_theta_zero_location('N')
        ax_hd_slow_all.set_theta_direction(-1)
        sig_slow_all = get_sig_stars(hd_slow_cell['p_value'])
        ax_hd_slow_all.set_title(f"HD All Positions (Stationary){sig_slow_all}\nPeak: {hd_slow_cell['peak_rate']:.1f} Hz, MVL: {hd_slow_cell['mvl']:.3f}, p={hd_slow_cell['p_value']:.3f}", fontsize=8, pad=10)
        
        plt.suptitle(f"Cell {cell_id} - Place Field & Head Direction Analysis", fontsize=10, fontweight='bold', y=0.98)
        
        if save_folder:
            if has_pf_run:
                prefix = "PlaceCell"
            else:
                prefix = "NonPlace"
            fname = f"{prefix}_cell{cell_id}_combined.pdf"
            plt.savefig(os.path.join(save_folder, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    print(f"Saved {n_cells} combined plots to {save_folder}")

# Combined Place Field + Head Direction Analysis - LARGEST PLACE FIELD ONLY

def get_largest_place_field_mask(place_field_mask):
    """
    Extract only the largest connected component from a place field mask.
    
    Parameters:
    -----------
    place_field_mask : 2D bool array
        Binary mask with potentially multiple place fields
        
    Returns:
    --------
    2D bool array with only the largest place field
    """
    if not np.any(place_field_mask):
        return place_field_mask.copy()
    
    # Label connected components
    labeled_mask, num_features = label(place_field_mask)
    
    if num_features == 0:
        return place_field_mask.copy()
    
    # Find the largest component
    largest_size = 0
    largest_label = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled_mask == i)
        if size > largest_size:
            largest_size = size
            largest_label = i
    
    # Create mask with only the largest field
    largest_mask = (labeled_mask == largest_label)
    return largest_mask


def plot_combined_pf_hd_largest(results_run, results_slow, hd_results_run, hd_results_slow,
                                 x_neural, y_neural, hd_angles_neural, spikes, 
                                 idx_run, idx_slow, bins, extent, frame_rate,
                                 save_folder=None):
    """
    Create combined PDF for each cell with:
    - Row 0: Place field rate maps (Running | Stationary) - showing LARGEST field only
    - Row 1: Trajectory + spikes colored by head direction (Running | Stationary)
    - Row 2: HD In Place Field (Running | Stationary) - LARGEST field only
    - Row 3: HD Outside Place Field (Running | Stationary) - LARGEST field only
    - Row 4: HD All Positions (Running | Stationary)
    
    Only considers the LARGEST place field for in/out of field HD analysis.
    """
    if save_folder:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    
    bin_centers_rad = np.deg2rad(hd_results_run[0]['bin_centers'])
    n_cells = len(results_run['results'])
    
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='white')
    
    for cell_idx in range(n_cells):
        res_run = results_run['results'][cell_idx]
        res_slow = results_slow['results'][cell_idx]
        cell_id = res_run['cell_id']
        
        # Check if cell has significant place field in either condition
        has_pf_run = res_run['p_value'] < 0.05 and np.any(res_run['place_field_mask'])
        has_pf_slow = res_slow['p_value'] < 0.05 and np.any(res_slow['place_field_mask'])
        has_any_pf = has_pf_run or has_pf_slow
        
        # Get LARGEST place field masks
        largest_pf_run = get_largest_place_field_mask(res_run['place_field_mask']) if has_pf_run else res_run['place_field_mask']
        largest_pf_slow = get_largest_place_field_mask(res_slow['place_field_mask']) if has_pf_slow else res_slow['place_field_mask']
        
        # Count number of fields for title info
        _, num_fields_run = label(res_run['place_field_mask'])
        _, num_fields_slow = label(res_slow['place_field_mask'])
        
        # Determine figure layout - always 5 rows for consistent layout
        fig = plt.figure(figsize=(8, 16))
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 0.9, 0.9, 0.9], hspace=0.5, wspace=0.3)
        
        # =====================================================================
        # ROW 0: PLACE FIELD RATE MAPS
        # =====================================================================
        
        # Running - Rate Map
        ax_pf_run = fig.add_subplot(gs[0, 0])
        map_run = res_run['rate_map']
        masked_map_run = ma.masked_where(np.isnan(map_run), map_run)
        im1 = ax_pf_run.imshow(masked_map_run.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        
        fields_info_run = f" ({num_fields_run} fields)" if num_fields_run > 1 else ""
        ax_pf_run.set_title(f"Cell {cell_id} - Running (≥2 cm/s){fields_info_run}\nPeak: {res_run['peak_rate']:.1f} Hz, SI: {res_run['si']:.2f}, p={res_run['p_value']:.3f}", fontsize=8)
        ax_pf_run.set_aspect('equal')
        ax_pf_run.set_xticks([])
        ax_pf_run.set_yticks([])
        
        div1 = make_axes_locatable(ax_pf_run)
        cax1 = div1.append_axes("right", size="5%", pad=0.05)
        cb1 = plt.colorbar(im1, cax=cax1)
        cb1.ax.tick_params(labelsize=6)
        
        # Draw ONLY the largest place field contour
        if has_pf_run:
            nx, ny = largest_pf_run.shape
            bin_x = (extent[1] - extent[0]) / nx
            bin_y = (extent[3] - extent[2]) / ny
            
            padded_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
            padded_mask[1:-1, 1:-1] = largest_pf_run
            padded_extent = (extent[0] - bin_x, extent[1] + bin_x, extent[2] - bin_y, extent[3] + bin_y)
            ax_pf_run.contour(padded_mask.T, levels=[0.5], colors='magenta', linewidths=1.5, extent=padded_extent, origin='lower')
            
            # Add "L" label at center of mass
            com = center_of_mass(largest_pf_run)
            if com:
                center_x = extent[0] + (com[0] * bin_x) + (bin_x / 2)
                center_y = extent[2] + (com[1] * bin_y) + (bin_y / 2)
                ax_pf_run.text(center_x, center_y, 'L', color='magenta', fontsize=8, 
                               ha='center', va='center', fontweight='bold', clip_on=True)
        
        # Stationary - Rate Map
        ax_pf_slow = fig.add_subplot(gs[0, 1])
        map_slow = res_slow['rate_map']
        masked_map_slow = ma.masked_where(np.isnan(map_slow), map_slow)
        im2 = ax_pf_slow.imshow(masked_map_slow.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        
        fields_info_slow = f" ({num_fields_slow} fields)" if num_fields_slow > 1 else ""
        ax_pf_slow.set_title(f"Cell {cell_id} - Stationary (<2 cm/s){fields_info_slow}\nPeak: {res_slow['peak_rate']:.1f} Hz, SI: {res_slow['si']:.2f}, p={res_slow['p_value']:.3f}", fontsize=8)
        ax_pf_slow.set_aspect('equal')
        ax_pf_slow.set_xticks([])
        ax_pf_slow.set_yticks([])
        
        div2 = make_axes_locatable(ax_pf_slow)
        cax2 = div2.append_axes("right", size="5%", pad=0.05)
        cb2 = plt.colorbar(im2, cax=cax2)
        cb2.ax.tick_params(labelsize=6)
        
        if has_pf_slow:
            nx, ny = largest_pf_slow.shape
            bin_x = (extent[1] - extent[0]) / nx
            bin_y = (extent[3] - extent[2]) / ny
            
            padded_mask = np.zeros((nx + 2, ny + 2), dtype=bool)
            padded_mask[1:-1, 1:-1] = largest_pf_slow
            padded_extent = (extent[0] - bin_x, extent[1] + bin_x, extent[2] - bin_y, extent[3] + bin_y)
            ax_pf_slow.contour(padded_mask.T, levels=[0.5], colors='magenta', linewidths=1.5, extent=padded_extent, origin='lower')
            
            com = center_of_mass(largest_pf_slow)
            if com:
                center_x = extent[0] + (com[0] * bin_x) + (bin_x / 2)
                center_y = extent[2] + (com[1] * bin_y) + (bin_y / 2)
                ax_pf_slow.text(center_x, center_y, 'L', color='magenta', fontsize=8, 
                               ha='center', va='center', fontweight='bold', clip_on=True)
        
        # =====================================================================
        # ROW 1: TRAJECTORY + SPIKES (colored by head direction)
        # =====================================================================
        
        # Get spike indices and corresponding HD angles for this cell
        cell_spike_indices = spikes[cell_idx]
        total_frames = len(hd_angles_neural)
        
        # Running - get spike HD angles
        binary_train_full_run = np.zeros(total_frames, dtype=bool)
        binary_train_full_run[cell_spike_indices] = True
        binary_train_run = binary_train_full_run[idx_run]
        spike_indices_run = np.where(binary_train_run)[0]
        hd_at_spikes_run = hd_angles_neural[idx_run][spike_indices_run]
        
        # Normalize HD to 0-1 for colormap
        hd_colors_run = hd_at_spikes_run / 360.0
        
        # Running - Trajectory with spikes colored by HD
        ax_traj_run = fig.add_subplot(gs[1, 0])
        ax_traj_run.plot(results_run['x_traj'], results_run['y_traj'], color='#cccccc', alpha=0.5, linewidth=0.3)
        if len(res_run['spikes_x']) > 0:
            ax_traj_run.scatter(res_run['spikes_x'], res_run['spikes_y'], s=3, c=hd_colors_run, 
                               cmap='hsv', alpha=0.8, linewidths=0, vmin=0, vmax=1)
        ax_traj_run.set_xlim(extent[0], extent[1])
        ax_traj_run.set_ylim(extent[2], extent[3])
        ax_traj_run.set_aspect('equal')
        ax_traj_run.set_xticks([])
        ax_traj_run.set_yticks([])
        ax_traj_run.set_title("Trajectory + Spikes by HD (Running)", fontsize=8)
        
        # Add circular colorbar for running
        add_circular_colorbar(fig, ax_traj_run)
        
        # Stationary - get spike HD angles
        binary_train_slow = binary_train_full_run[idx_slow]
        spike_indices_slow = np.where(binary_train_slow)[0]
        hd_at_spikes_slow = hd_angles_neural[idx_slow][spike_indices_slow]
        
        # Normalize HD to 0-1 for colormap
        hd_colors_slow = hd_at_spikes_slow / 360.0
        
        # Stationary - Trajectory with spikes colored by HD
        ax_traj_slow = fig.add_subplot(gs[1, 1])
        ax_traj_slow.plot(results_slow['x_traj'], results_slow['y_traj'], color='#cccccc', alpha=0.5, linewidth=0.3)
        if len(res_slow['spikes_x']) > 0:
            ax_traj_slow.scatter(res_slow['spikes_x'], res_slow['spikes_y'], s=3, c=hd_colors_slow, 
                                cmap='hsv', alpha=0.8, linewidths=0, vmin=0, vmax=1)
        ax_traj_slow.set_xlim(extent[0], extent[1])
        ax_traj_slow.set_ylim(extent[2], extent[3])
        ax_traj_slow.set_aspect('equal')
        ax_traj_slow.set_xticks([])
        ax_traj_slow.set_yticks([])
        ax_traj_slow.set_title("Trajectory + Spikes by HD (Stationary)", fontsize=8)
        
        # Add circular colorbar for stationary
        add_circular_colorbar(fig, ax_traj_slow)
        
        # =====================================================================
        # HD ANALYSIS - Using LARGEST place field only
        # =====================================================================
        
        theta_closed = np.append(bin_centers_rad, bin_centers_rad[0])
        
        # Helper function to add scaled occupancy probability overlay
        def add_occupancy_overlay(ax, occ_prob, theta_closed, peak_rate):
            """Add gray occupancy probability overlay scaled to firing rate axis."""
            occ_closed = np.append(occ_prob, occ_prob[0])
            if peak_rate > 0:
                occ_scaled = occ_closed * (peak_rate * 0.5) / np.max(occ_closed) if np.max(occ_closed) > 0 else occ_closed
            else:
                occ_scaled = occ_closed
            ax.fill(theta_closed, occ_scaled, alpha=0.15, color='gray', zorder=0)
            ax.plot(theta_closed, occ_scaled, color='gray', linewidth=0.8, alpha=0.5, zorder=0)
        
        # Helper function for significance stars
        def get_sig_stars(p_value):
            """Return significance stars based on p-value."""
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ''
        
        # === RUNNING HD Analysis (Left Column) - LARGEST PF ONLY ===
        if has_pf_run:
            x_run = x_neural[idx_run]
            y_run = y_neural[idx_run]
            # Use LARGEST place field mask
            in_field_run = get_in_place_field_indices(x_run, y_run, largest_pf_run, bins)
            idx_run_in = idx_run[in_field_run]
            idx_run_out = idx_run[~in_field_run]
            
            hd_run_in = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_run_in, frame_rate)[0]
            hd_run_out = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_run_out, frame_rate)[0]
            
            # ROW 2 Left: Running In Largest Place Field
            ax_hd_run_in = fig.add_subplot(gs[2, 0], projection='polar')
            add_occupancy_overlay(ax_hd_run_in, hd_run_in['hd_occupancy_prob'], theta_closed, hd_run_in['peak_rate'])
            fr_run_in = np.append(hd_run_in['firing_rate'], hd_run_in['firing_rate'][0])
            ax_hd_run_in.plot(theta_closed, fr_run_in, 'b-', linewidth=1.5, zorder=2)
            ax_hd_run_in.fill(theta_closed, fr_run_in, alpha=0.3, color='blue', zorder=1)
            ax_hd_run_in.set_theta_zero_location('N')
            ax_hd_run_in.set_theta_direction(-1)
            sig_run_in = get_sig_stars(hd_run_in['p_value'])
            ax_hd_run_in.set_title(f"HD In Largest PF (Running){sig_run_in}\nPeak: {hd_run_in['peak_rate']:.1f} Hz, MVL: {hd_run_in['mvl']:.3f}, p={hd_run_in['p_value']:.3f}", fontsize=8, pad=10)
            
            # ROW 3 Left: Running Outside Largest Place Field
            ax_hd_run_out = fig.add_subplot(gs[3, 0], projection='polar')
            add_occupancy_overlay(ax_hd_run_out, hd_run_out['hd_occupancy_prob'], theta_closed, hd_run_out['peak_rate'])
            fr_run_out = np.append(hd_run_out['firing_rate'], hd_run_out['firing_rate'][0])
            ax_hd_run_out.plot(theta_closed, fr_run_out, 'purple', linewidth=1.5, zorder=2)
            ax_hd_run_out.fill(theta_closed, fr_run_out, alpha=0.3, color='purple', zorder=1)
            ax_hd_run_out.set_theta_zero_location('N')
            ax_hd_run_out.set_theta_direction(-1)
            sig_run_out = get_sig_stars(hd_run_out['p_value'])
            ax_hd_run_out.set_title(f"HD Outside Largest PF (Running){sig_run_out}\nPeak: {hd_run_out['peak_rate']:.1f} Hz, MVL: {hd_run_out['mvl']:.3f}, p={hd_run_out['p_value']:.3f}", fontsize=8, pad=10)
        
        # === STATIONARY HD Analysis (Right Column) - LARGEST PF ONLY ===
        if has_pf_slow:
            x_slow = x_neural[idx_slow]
            y_slow = y_neural[idx_slow]
            # Use LARGEST place field mask
            in_field_slow = get_in_place_field_indices(x_slow, y_slow, largest_pf_slow, bins)
            idx_slow_in = idx_slow[in_field_slow]
            idx_slow_out = idx_slow[~in_field_slow]
            
            hd_slow_in = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_slow_in, frame_rate)[0]
            hd_slow_out = analyze_hd_simple(hd_angles_neural, [spikes[cell_idx]], idx_slow_out, frame_rate)[0]
            
            # ROW 2 Right: Stationary In Largest Place Field
            ax_hd_slow_in = fig.add_subplot(gs[2, 1], projection='polar')
            add_occupancy_overlay(ax_hd_slow_in, hd_slow_in['hd_occupancy_prob'], theta_closed, hd_slow_in['peak_rate'])
            fr_slow_in = np.append(hd_slow_in['firing_rate'], hd_slow_in['firing_rate'][0])
            ax_hd_slow_in.plot(theta_closed, fr_slow_in, 'g-', linewidth=1.5, zorder=2)
            ax_hd_slow_in.fill(theta_closed, fr_slow_in, alpha=0.3, color='green', zorder=1)
            ax_hd_slow_in.set_theta_zero_location('N')
            ax_hd_slow_in.set_theta_direction(-1)
            sig_slow_in = get_sig_stars(hd_slow_in['p_value'])
            ax_hd_slow_in.set_title(f"HD In Largest PF (Stationary){sig_slow_in}\nPeak: {hd_slow_in['peak_rate']:.1f} Hz, MVL: {hd_slow_in['mvl']:.3f}, p={hd_slow_in['p_value']:.3f}", fontsize=8, pad=10)
            
            # ROW 3 Right: Stationary Outside Largest Place Field
            ax_hd_slow_out = fig.add_subplot(gs[3, 1], projection='polar')
            add_occupancy_overlay(ax_hd_slow_out, hd_slow_out['hd_occupancy_prob'], theta_closed, hd_slow_out['peak_rate'])
            fr_slow_out = np.append(hd_slow_out['firing_rate'], hd_slow_out['firing_rate'][0])
            ax_hd_slow_out.plot(theta_closed, fr_slow_out, 'orange', linewidth=1.5, zorder=2)
            ax_hd_slow_out.fill(theta_closed, fr_slow_out, alpha=0.3, color='orange', zorder=1)
            ax_hd_slow_out.set_theta_zero_location('N')
            ax_hd_slow_out.set_theta_direction(-1)
            sig_slow_out = get_sig_stars(hd_slow_out['p_value'])
            ax_hd_slow_out.set_title(f"HD Outside Largest PF (Stationary){sig_slow_out}\nPeak: {hd_slow_out['peak_rate']:.1f} Hz, MVL: {hd_slow_out['mvl']:.3f}, p={hd_slow_out['p_value']:.3f}", fontsize=8, pad=10)
        
        # === ROW 4: HD All Positions (always plotted) ===
        hd_run_cell = hd_results_run[cell_idx]
        hd_slow_cell = hd_results_slow[cell_idx]
        
        # Left: Running All Positions
        ax_hd_run_all = fig.add_subplot(gs[4, 0], projection='polar')
        add_occupancy_overlay(ax_hd_run_all, hd_run_cell['occupancy'] / np.sum(hd_run_cell['occupancy']) if np.sum(hd_run_cell['occupancy']) > 0 else hd_run_cell['occupancy'], theta_closed, hd_run_cell['peak_rate'])
        fr_run_all = np.append(hd_run_cell['firing_rate'], hd_run_cell['firing_rate'][0])
        ax_hd_run_all.plot(theta_closed, fr_run_all, 'darkblue', linewidth=1.5, zorder=2)
        ax_hd_run_all.fill(theta_closed, fr_run_all, alpha=0.3, color='darkblue', zorder=1)
        ax_hd_run_all.set_theta_zero_location('N')
        ax_hd_run_all.set_theta_direction(-1)
        sig_run_all = get_sig_stars(hd_run_cell['p_value'])
        ax_hd_run_all.set_title(f"HD All Positions (Running){sig_run_all}\nPeak: {hd_run_cell['peak_rate']:.1f} Hz, MVL: {hd_run_cell['mvl']:.3f}, p={hd_run_cell['p_value']:.3f}", fontsize=8, pad=10)
        
        # Right: Stationary All Positions
        ax_hd_slow_all = fig.add_subplot(gs[4, 1], projection='polar')
        add_occupancy_overlay(ax_hd_slow_all, hd_slow_cell['occupancy'] / np.sum(hd_slow_cell['occupancy']) if np.sum(hd_slow_cell['occupancy']) > 0 else hd_slow_cell['occupancy'], theta_closed, hd_slow_cell['peak_rate'])
        fr_slow_all = np.append(hd_slow_cell['firing_rate'], hd_slow_cell['firing_rate'][0])
        ax_hd_slow_all.plot(theta_closed, fr_slow_all, 'darkgreen', linewidth=1.5, zorder=2)
        ax_hd_slow_all.fill(theta_closed, fr_slow_all, alpha=0.3, color='darkgreen', zorder=1)
        ax_hd_slow_all.set_theta_zero_location('N')
        ax_hd_slow_all.set_theta_direction(-1)
        sig_slow_all = get_sig_stars(hd_slow_cell['p_value'])
        ax_hd_slow_all.set_title(f"HD All Positions (Stationary){sig_slow_all}\nPeak: {hd_slow_cell['peak_rate']:.1f} Hz, MVL: {hd_slow_cell['mvl']:.3f}, p={hd_slow_cell['p_value']:.3f}", fontsize=8, pad=10)
        
        plt.suptitle(f"Cell {cell_id} - Place Field & HD Analysis (Largest PF Only)", fontsize=10, fontweight='bold', y=0.98)
        
        if save_folder:
            if has_pf_run:
                prefix = "PlaceCell"
            else:
                prefix = "NonPlace"
            fname = f"{prefix}_cell{cell_id}_largest_PF.pdf"
            plt.savefig(os.path.join(save_folder, fname), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    print(f"Saved {n_cells} combined plots (largest PF only) to {save_folder}")

def batch_place_cell_head_direction_analysis(x_neural, y_neural, hd_angles_neural, spikes, speed, traces, frame_rate, figure_folder, 
                              width_real = 35.5, height_real = 20, bin_size = 1.5, 
                              speed_threshold = 2, place_field_threshold = 0.4, min_field_bins = 5):

    arena_size = (width_real, height_real)
    #arena_size = (np.ceil(np.max(x_neural)), np.ceil(np.max(y_neural)))
    bins = [np.arange(0, arena_size[0] + bin_size, bin_size),
            np.arange(0, arena_size[1] + bin_size, bin_size)]
    extent = (0, width_real, 0, height_real)

    # 2. Define Indices
    # ==== 1. GLOBAL NaN HANDLING ====
    # Any frame where x_neural, y_neural or speed is NaN will be excluded from *all* analyses
    valid_frames = (~np.isnan(x_neural)) & (~np.isnan(y_neural)) & (~np.isnan(speed)) & (~np.isnan(traces).any(axis=0))

    n_total = len(x_neural)
    n_valid = np.sum(valid_frames)
    n_nan = n_total - n_valid

    print(f"Total frames: {n_total}")
    print(f"Valid frames (no NaN in x/y/speed): {n_valid}")
    print(f"Frames excluded due to NaN: {n_nan} ({100.0 * n_nan / n_total:.2f}%)")

    # ---- Simple version: no minimum epoch length ----
    idx_run = np.where((speed >= speed_threshold) & valid_frames)[0]
    idx_slow = np.where((speed < speed_threshold) & valid_frames)[0]

    print("Running Analysis for Speed >= 2...")
    results_run = analyze_place_cell_subset(
        x_neural, y_neural, spikes, 
        subset_indices=idx_run, 
        bins=bins, frame_rate=frame_rate, 
        place_field_threshold=place_field_threshold,
        min_field_bins=min_field_bins
    )

    # 4. RUN 2: SLOW / STATIONARY
    print("Running Analysis for Speed < 2...")
    results_slow = analyze_place_cell_subset(
        x_neural, y_neural, spikes, 
        subset_indices=idx_slow, 
        bins=bins, frame_rate=frame_rate, 
        place_field_threshold=place_field_threshold,
        min_field_bins=min_field_bins
    )


    hd_results_run = analyze_head_direction(
        hd_angles_neural, spikes, 
        subset_indices=idx_run, 
        frame_rate=frame_rate,
        bin_size_deg=15,
        num_shuffles=1000
    )

    print("Running Head Direction Analysis for Speed < 2...")
    hd_results_slow = analyze_head_direction(
        hd_angles_neural, spikes, 
        subset_indices=idx_slow, 
        frame_rate=frame_rate,
        bin_size_deg=15,
        num_shuffles=1000
    )

    hd_run_list = hd_results_run['results']
    hd_slow_list = hd_results_slow['results']

    plot_combined_pf_hd_largest(
        results_run, results_slow, 
        hd_run_list, hd_slow_list,
        x_neural, y_neural, hd_angles_neural, spikes,
        idx_run, idx_slow, bins, extent, frame_rate,
        save_folder=figure_folder
    )

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y