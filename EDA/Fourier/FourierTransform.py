"""
FFT-based time series decomposition for identifying dominant frequency components.

Uses Fast Fourier Transform to decompose ED volume time series into frequency components,
identifying periodicities (daily, weekly, yearly) and reconstructing signals as sums
of sine/cosine waves. Useful when traditional decomposition gives unclear patterns.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from typing import Optional, Tuple, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from EDA_Data_Loader import load_dataset

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Use non-interactive backend for batch processing
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - no plt.show() needed


def aggregate_to_daily(df: pd.DataFrame, 
                       site: Optional[str] = None,
                       block: Optional[int] = None) -> pd.Series:
    """
    Aggregate data to daily time series.
    
    Args:
        df: Raw dataframe with Date, Site, Hour, ED Enc columns
        site: Optional site filter (A, B, C, D)
        block: Optional block filter (0-3)
    
    Returns:
        Series with Date index and daily ED Enc values
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Apply filters
    if site:
        df = df[df['Site'] == site]
    if block is not None:
        df['Block'] = (df['Hour'] // 6).astype(int)
        df = df[df['Block'] == block]
    
    # Aggregate to daily
    daily = df.groupby('Date')['ED Enc'].sum().sort_index()
    
    # Fill missing dates with 0 (or interpolate if preferred)
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(date_range, fill_value=0)
    
    return daily


def aggregate_to_block_daily(df: pd.DataFrame,
                             site: Optional[str] = None,
                             block: int = 0) -> pd.Series:
    """
    Aggregate to daily time series for a specific block.
    
    Args:
        df: Raw dataframe
        site: Optional site filter
        block: Block number (0-3)
    
    Returns:
        Series with Date index and daily ED Enc values for the block
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Block'] = (df['Hour'] // 6).astype(int)
    
    if site:
        df = df[df['Site'] == site]
    df = df[df['Block'] == block]
    
    daily = df.groupby('Date')['ED Enc'].sum().sort_index()
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(date_range, fill_value=0)
    
    return daily


def perform_fft_analysis(ts: pd.Series, 
                        sampling_rate: float = 1.0,
                        detrend: bool = True) -> Dict:
    """
    Perform FFT analysis on time series.
    
    Args:
        ts: Time series (must be regularly spaced)
        sampling_rate: Samples per unit time (1.0 for daily = 1 sample/day)
        detrend: Whether to remove linear trend before FFT
    
    Returns:
        Dictionary with FFT results:
        - frequencies: Frequency values (cycles per unit time)
        - fft_values: Complex FFT coefficients
        - power_spectrum: Power spectral density
        - periods: Periods in days (for daily sampling)
        - dominant_freqs: Top N dominant frequencies
    """
    values = ts.values
    n = len(values)
    
    # Detrend if requested
    if detrend:
        x = np.arange(n)
        trend = np.polyfit(x, values, 1)
        values = values - np.polyval(trend, x)
    
    # Remove mean
    values = values - values.mean()
    
    # Compute FFT
    fft_vals = fft(values)
    freqs = fftfreq(n, d=1/sampling_rate)
    
    # Only positive frequencies (first half)
    positive_freq_idx = freqs > 0
    freqs_positive = freqs[positive_freq_idx]
    fft_positive = fft_vals[positive_freq_idx]
    
    # Power spectrum (magnitude squared)
    power = np.abs(fft_positive) ** 2
    
    # Periods (for daily sampling, period = 1/frequency in days)
    periods = 1.0 / freqs_positive
    
    # Find dominant frequencies (top peaks)
    # Use scipy's find_peaks for robust peak detection
    # Normalize power for peak detection
    power_normalized = power / power.max()
    min_distance = max(1, len(power) // 100)  # Ensure distance >= 1
    peaks, properties = signal.find_peaks(
        power_normalized,
        height=0.1,  # At least 10% of max power
        distance=min_distance  # Minimum distance between peaks
    )
    
    # Sort peaks by power
    peak_powers = power[peaks]
    peak_indices = peaks[np.argsort(peak_powers)[::-1]]
    
    # Get top 20 dominant frequencies
    top_n = min(20, len(peak_indices))
    dominant_indices = peak_indices[:top_n]
    
    dominant_freqs = pd.DataFrame({
        'frequency': freqs_positive[dominant_indices],
        'period_days': periods[dominant_indices],
        'power': power[dominant_indices],
        'amplitude': np.abs(fft_positive[dominant_indices]),
        'phase': np.angle(fft_positive[dominant_indices])
    }).sort_values('power', ascending=False)
    
    return {
        'frequencies': freqs_positive,
        'fft_values': fft_positive,
        'power_spectrum': power,
        'periods': periods,
        'dominant_freqs': dominant_freqs,
        'detrended_values': values,
        'n_samples': n
    }


def identify_known_periods(dominant_freqs: pd.DataFrame,
                           expected_periods: List[float] = [1.0, 7.0, 365.25],
                           tolerance: float = 0.1) -> pd.DataFrame:
    """
    Identify which dominant frequencies correspond to known periods.
    
    Args:
        dominant_freqs: DataFrame with period_days column
        expected_periods: List of expected periods in days
        tolerance: Relative tolerance for matching (0.1 = 10%)
    
    Returns:
        DataFrame with matched_period column indicating which known period matches
    """
    result = dominant_freqs.copy()
    result['matched_period'] = None
    result['period_name'] = None
    
    period_names = {
        1.0: 'Daily',
        4.0: 'Block (4 blocks/day)',
        7.0: 'Weekly',
        30.0: 'Monthly',
        365.25: 'Yearly',
        182.625: 'Semi-annual'
    }
    
    for idx, row in result.iterrows():
        period = row['period_days']
        for exp_period in expected_periods:
            if abs(period - exp_period) / exp_period <= tolerance:
                result.at[idx, 'matched_period'] = exp_period
                result.at[idx, 'period_name'] = period_names.get(exp_period, f'{exp_period:.1f} days')
                break
    
    return result


def reconstruct_signal(fft_results: Dict, 
                      n_components: int = 10,
                      original_mean: float = 0.0) -> np.ndarray:
    """
    Reconstruct signal using top N frequency components.
    
    Args:
        fft_results: Results from perform_fft_analysis
        n_components: Number of top components to use
        original_mean: Mean of original signal (to add back)
    
    Returns:
        Reconstructed signal values
    """
    fft_vals = fft_results['fft_values']
    n = fft_results['n_samples']
    
    # Get top N components by power
    dominant = fft_results['dominant_freqs'].head(n_components)
    
    # Create zero array for reconstruction
    fft_reconstructed = np.zeros(n, dtype=complex)
    
    # Add DC component (mean)
    fft_reconstructed[0] = original_mean * n
    
    # Add top N components (and their negative frequency counterparts)
    freq_indices = np.searchsorted(fft_results['frequencies'], dominant['frequency'].values)
    
    for i, freq_idx in enumerate(freq_indices):
        # Positive frequency component
        if freq_idx < len(fft_vals):
            fft_reconstructed[freq_idx] = fft_vals[freq_idx]
        # Negative frequency component (symmetric)
        neg_idx = n - freq_idx
        if neg_idx < n:
            fft_reconstructed[neg_idx] = np.conj(fft_vals[freq_idx])
    
    # Inverse FFT
    reconstructed = np.real(ifft(fft_reconstructed))
    
    return reconstructed


def plot_fft_analysis(ts: pd.Series,
                     fft_results: Dict,
                     site: Optional[str] = None,
                     block: Optional[int] = None,
                     save_path: Optional[Path] = None):
    """
    Create comprehensive FFT analysis visualization.
    
    Args:
        ts: Original time series
        fft_results: Results from perform_fft_analysis
        site: Site identifier for title
        block: Block identifier for title
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Title
    title_parts = ['FFT Analysis']
    if site:
        title_parts.append(f'Site {site}')
    if block is not None:
        title_parts.append(f'Block {block}')
    title = ' - '.join(title_parts)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Original time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts.index, ts.values, 'b-', linewidth=1, alpha=0.7, label='Original')
    if 'detrended_values' in fft_results:
        detrended = fft_results['detrended_values']
        reconstructed_detrended = reconstruct_signal(fft_results, n_components=10, original_mean=0.0)
        # Add back original mean and trend
        reconstructed_full = reconstructed_detrended + ts.values.mean()
        ax1.plot(ts.index, reconstructed_full, 'r--', linewidth=2, alpha=0.8, label='Reconstructed (top 10 components)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('ED Encounters')
    ax1.set_title('Time Series: Original vs Reconstructed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Frequency spectrum (linear scale)
    ax2 = fig.add_subplot(gs[1, 0])
    periods = fft_results['periods']
    power = fft_results['power_spectrum']
    
    # Focus on periods from 0.5 days to 2000 days
    mask = (periods >= 0.5) & (periods <= 2000)
    ax2.plot(periods[mask], power[mask], 'b-', linewidth=1.5)
    ax2.set_xlabel('Period (days)')
    ax2.set_ylabel('Power')
    ax2.set_title('Power Spectrum (Linear Scale)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Mark known periods
    known_periods = [1, 7, 30, 365.25]
    for period in known_periods:
        if period >= periods[mask].min() and period <= periods[mask].max():
            ax2.axvline(period, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax2.text(period, ax2.get_ylim()[1] * 0.9, f'{period:.1f}d', 
                    rotation=90, ha='right', fontsize=8)
    
    # Mark top peaks
    top_freqs = fft_results['dominant_freqs'].head(10)
    for _, row in top_freqs.iterrows():
        period = row['period_days']
        if mask.sum() > 0 and period >= periods[mask].min() and period <= periods[mask].max():
            idx = np.argmin(np.abs(periods[mask] - period))
            power_val = power[mask][idx]
            ax2.plot(period, power_val, 'ro', markersize=8, alpha=0.7)
    
    # 3. Frequency spectrum (log scale, zoomed)
    ax3 = fig.add_subplot(gs[1, 1])
    # Focus on shorter periods (daily to yearly)
    mask_short = (periods >= 0.5) & (periods <= 500)
    ax3.plot(periods[mask_short], power[mask_short], 'b-', linewidth=1.5)
    ax3.set_xlabel('Period (days)')
    ax3.set_ylabel('Power (log scale)')
    ax3.set_title('Power Spectrum (Short Periods)')
    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Mark known periods
    for period in known_periods:
        if period >= periods[mask_short].min() and period <= periods[mask_short].max():
            ax3.axvline(period, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax3.text(period, ax3.get_ylim()[1] * 0.5, f'{period:.1f}d', 
                    rotation=90, ha='right', fontsize=8)
    
    # 4. Top dominant frequencies table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    top_freqs_display = fft_results['dominant_freqs'].head(15).copy()
    top_freqs_display['period_days'] = top_freqs_display['period_days'].round(2)
    top_freqs_display['power'] = top_freqs_display['power'].round(2)
    top_freqs_display['amplitude'] = top_freqs_display['amplitude'].round(2)
    
    # Identify known periods
    top_freqs_display = identify_known_periods(top_freqs_display)
    
    # Add period name column for display
    display_cols = ['period_days', 'frequency', 'power', 'amplitude', 'period_name']
    table_data = top_freqs_display[display_cols].copy()
    table_data.columns = ['Period (days)', 'Frequency', 'Power', 'Amplitude', 'Period Name']
    table_data = table_data.fillna('')
    
    table = ax4.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color known periods
    for i in range(len(table_data)):
        if table_data.iloc[i]['Period Name']:
            for j in range(len(table_data.columns)):
                table[(i+1, j)].set_facecolor('#e6f3ff')
    
    ax4.set_title('Top 15 Dominant Frequency Components', fontsize=12, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    plt.close(fig)  # Close to free memory


def perform_multiresolution_fft(df: pd.DataFrame,
                                 site: Optional[str] = None,
                                 resolutions: List[str] = ['D', 'W', 'ME']) -> Dict[str, Dict]:
    """
    Perform FFT analysis at multiple time resolutions to reveal patterns at different scales.
    
    Higher resolutions (daily) capture short-term cycles (weekly).
    Lower resolutions (weekly/monthly) reduce noise and reveal longer-term patterns.
    
    Args:
        df: Raw dataframe with Date, Site, Hour, ED Enc columns
        site: Optional site filter
        resolutions: List of pandas frequency strings ('D'=daily, 'W'=weekly, 'M'=monthly)
    
    Returns:
        Dictionary mapping resolution -> FFT results
    """
    results = {}
    
    resolution_names = {
        'D': 'Daily',
        'W': 'Weekly', 
        'ME': 'Monthly',
        'M': 'Monthly',  # Legacy support
        '2W': 'Bi-weekly',
        'QE': 'Quarterly',
        'Q': 'Quarterly'  # Legacy support
    }
    
    # Prepare base data
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    if site:
        df = df[df['Site'] == site]
    
    # Daily aggregation first
    daily = df.groupby('Date')['ED Enc'].sum().sort_index()
    date_range = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(date_range, fill_value=0)
    
    for res in resolutions:
        res_name = resolution_names.get(res, res)
        print(f"  Processing {res_name} resolution...")
        
        if res == 'D':
            ts = daily
            sampling_period_days = 1
        else:
            # Resample to lower resolution
            ts = daily.resample(res).sum()
            # Calculate sampling period in days
            if res == 'W':
                sampling_period_days = 7
            elif res == '2W':
                sampling_period_days = 14
            elif res in ('M', 'ME'):
                sampling_period_days = 30.44  # Average month length
            elif res in ('Q', 'QE'):
                sampling_period_days = 91.31  # Average quarter length
            else:
                sampling_period_days = 1
        
        if len(ts) < 10:
            print(f"    Skipping {res_name}: insufficient data points ({len(ts)})")
            continue
        
        # Perform FFT (sampling_rate = samples per day)
        sampling_rate = 1.0 / sampling_period_days
        fft_results = perform_fft_analysis(ts, sampling_rate=sampling_rate, detrend=True)
        
        # Convert periods back to days for comparison
        fft_results['periods_days'] = fft_results['periods'] * sampling_period_days
        fft_results['dominant_freqs']['period_days_actual'] = (
            fft_results['dominant_freqs']['period_days'] * sampling_period_days
        )
        
        results[res] = {
            'name': res_name,
            'time_series': ts,
            'fft_results': fft_results,
            'sampling_period_days': sampling_period_days,
            'n_samples': len(ts)
        }
    
    return results


def plot_multiresolution_fft(multiresolution_results: Dict[str, Dict],
                              site: Optional[str] = None,
                              save_path: Optional[Path] = None):
    """
    Create comprehensive multi-resolution FFT visualization.
    
    Args:
        multiresolution_results: Results from perform_multiresolution_fft
        site: Site identifier for title
        save_path: Optional path to save figure
    """
    n_resolutions = len(multiresolution_results)
    if n_resolutions == 0:
        print("No results to plot")
        return
    
    fig = plt.figure(figsize=(18, 5 * n_resolutions + 4))
    
    # Title
    title = 'Multi-Resolution FFT Analysis'
    if site:
        title += f' - Site {site}'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid: each resolution gets 2 rows (time series + spectrum)
    gs = fig.add_gridspec(n_resolutions * 2 + 1, 2, 
                          height_ratios=[1] * (n_resolutions * 2) + [1.5],
                          hspace=0.4, wspace=0.3)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_resolutions))
    
    # Summary data for comparison table
    summary_data = []
    
    for i, (res, data) in enumerate(multiresolution_results.items()):
        ts = data['time_series']
        fft_results = data['fft_results']
        res_name = data['name']
        color = colors[i]
        
        # Time series plot
        ax_ts = fig.add_subplot(gs[i * 2, :])
        ax_ts.plot(ts.index, ts.values, color=color, linewidth=1, alpha=0.8)
        
        # Add smoothed line for noisy data
        if len(ts) > 20:
            window = min(7, len(ts) // 5)
            smoothed = ts.rolling(window=window, center=True).mean()
            ax_ts.plot(ts.index, smoothed.values, color='black', linewidth=2, 
                      linestyle='--', alpha=0.6, label=f'{window}-point MA')
        
        ax_ts.set_ylabel('Volume')
        ax_ts.set_title(f'{res_name} Time Series (n={len(ts)})', fontsize=11, fontweight='bold')
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(loc='upper right', fontsize=8)
        
        # Power spectrum plot
        ax_psd = fig.add_subplot(gs[i * 2 + 1, 0])
        
        # Use periods in days for x-axis
        periods_days = data['fft_results']['periods_days']
        power = fft_results['power_spectrum']
        
        # Filter to reasonable period range
        min_period = data['sampling_period_days'] * 2  # Nyquist limit
        max_period = len(ts) * data['sampling_period_days'] / 2
        mask = (periods_days >= min_period) & (periods_days <= max_period)
        
        if mask.sum() > 0:
            ax_psd.plot(periods_days[mask], power[mask], color=color, linewidth=1.5)
            ax_psd.set_xscale('log')
            ax_psd.set_xlabel('Period (days)')
            ax_psd.set_ylabel('Power')
            ax_psd.set_title(f'{res_name} Power Spectrum', fontsize=10)
            ax_psd.grid(True, alpha=0.3)
            
            # Mark known periods
            known_periods = [7, 30, 90, 182.5, 365.25]
            for period in known_periods:
                if min_period <= period <= max_period:
                    ax_psd.axvline(period, color='red', linestyle='--', alpha=0.4, linewidth=1)
        
        # Top frequencies bar chart
        ax_bar = fig.add_subplot(gs[i * 2 + 1, 1])
        top_freqs = fft_results['dominant_freqs'].head(8)
        
        if 'period_days_actual' in top_freqs.columns:
            periods_to_plot = top_freqs['period_days_actual'].values
        else:
            periods_to_plot = top_freqs['period_days'].values * data['sampling_period_days']
        
        powers_to_plot = top_freqs['power'].values
        
        # Create labels
        labels = []
        for p in periods_to_plot:
            if p < 1:
                labels.append(f'{p*24:.1f}h')
            elif p < 14:
                labels.append(f'{p:.1f}d')
            elif p < 60:
                labels.append(f'{p/7:.1f}w')
            elif p < 400:
                labels.append(f'{p/30.44:.1f}mo')
            else:
                labels.append(f'{p/365.25:.1f}yr')
        
        bars = ax_bar.barh(range(len(labels)), powers_to_plot, color=color, alpha=0.7)
        ax_bar.set_yticks(range(len(labels)))
        ax_bar.set_yticklabels(labels)
        ax_bar.set_xlabel('Power')
        ax_bar.set_title(f'{res_name} Top 8 Periods', fontsize=10)
        ax_bar.invert_yaxis()
        ax_bar.grid(True, alpha=0.3, axis='x')
        
        # Collect summary data
        for j, (period, power_val) in enumerate(zip(periods_to_plot[:5], powers_to_plot[:5])):
            summary_data.append({
                'Resolution': res_name,
                'Rank': j + 1,
                'Period (days)': f'{period:.1f}',
                'Power': f'{power_val:.2e}',
                'Period Label': labels[j] if j < len(labels) else ''
            })
    
    # Summary comparison table
    ax_table = fig.add_subplot(gs[-1, :])
    ax_table.axis('off')
    
    # Create comparison DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Pivot for better comparison
        pivot_data = []
        for res, data in multiresolution_results.items():
            res_name = data['name']
            top_freqs = data['fft_results']['dominant_freqs'].head(5)
            
            if 'period_days_actual' in top_freqs.columns:
                periods = top_freqs['period_days_actual'].values
            else:
                periods = top_freqs['period_days'].values * data['sampling_period_days']
            
            row = {'Resolution': res_name, 'Samples': data['n_samples']}
            for j, p in enumerate(periods):
                if p < 14:
                    row[f'#{j+1}'] = f'{p:.1f}d'
                elif p < 60:
                    row[f'#{j+1}'] = f'{p/7:.1f}w'
                elif p < 400:
                    row[f'#{j+1}'] = f'{p/30.44:.1f}mo'
                else:
                    row[f'#{j+1}'] = f'{p/365.25:.1f}yr'
            pivot_data.append(row)
        
        table_df = pd.DataFrame(pivot_data)
        
        # Fill NaN with empty string
        table_df = table_df.fillna('')
        
        table = ax_table.table(cellText=table_df.values,
                               colLabels=table_df.columns,
                               cellLoc='center',
                               loc='center',
                               bbox=[0.1, 0.2, 0.8, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Header styling
        for j in range(len(table_df.columns)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        ax_table.set_title('Top 5 Dominant Periods by Resolution', fontsize=12, 
                          fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved multi-resolution FFT plot to: {save_path}")
    
    plt.close(fig)  # Close to free memory


def print_multiresolution_summary(multiresolution_results: Dict[str, Dict]):
    """Print comparative summary of multi-resolution FFT analysis."""
    print("\n" + "=" * 100)
    print("MULTI-RESOLUTION FFT ANALYSIS SUMMARY")
    print("=" * 100)
    
    for res, data in multiresolution_results.items():
        res_name = data['name']
        ts = data['time_series']
        fft_results = data['fft_results']
        
        print(f"\n{'-' * 100}")
        print(f"RESOLUTION: {res_name.upper()}")
        print(f"{'-' * 100}")
        print(f"  Samples: {len(ts)}")
        print(f"  Sampling period: {data['sampling_period_days']:.2f} days")
        print(f"  Date range: {ts.index.min()} to {ts.index.max()}")
        print(f"  Mean volume: {ts.mean():.2f}")
        print(f"  Std volume: {ts.std():.2f}")
        print(f"  CV (Std/Mean): {ts.std()/ts.mean():.2%}")
        
        print(f"\n  Top 10 Dominant Periods:")
        top_freqs = fft_results['dominant_freqs'].head(10).copy()
        
        # Use actual period in days
        if 'period_days_actual' in top_freqs.columns:
            top_freqs['period_days'] = top_freqs['period_days_actual']
        else:
            top_freqs['period_days'] = top_freqs['period_days'] * data['sampling_period_days']
        
        # Identify known periods (requires period_days column)
        top_freqs = identify_known_periods(top_freqs)
        top_freqs['Period (days)'] = top_freqs['period_days']
        
        display_df = top_freqs[['Period (days)', 'power', 'amplitude', 'period_name']].copy()
        display_df.columns = ['Period (days)', 'Power', 'Amplitude', 'Known Period']
        display_df['Period (days)'] = display_df['Period (days)'].round(2)
        display_df['Power'] = display_df['Power'].apply(lambda x: f'{x:.2e}')
        display_df['Amplitude'] = display_df['Amplitude'].round(2)
        display_df = display_df.fillna('')
        
        print(display_df.to_string(index=False))
        
        # Reconstruction quality
        reconstructed = reconstruct_signal(fft_results, n_components=10, 
                                          original_mean=ts.values.mean())
        r2 = 1 - (np.sum((ts.values - reconstructed) ** 2) / 
                  np.sum((ts.values - ts.values.mean()) ** 2))
        print(f"\n  Reconstruction R² (top 10 components): {r2:.4f} ({r2*100:.1f}% variance explained)")
    
    # Cross-resolution comparison
    print("\n" + "=" * 100)
    print("CROSS-RESOLUTION INSIGHTS")
    print("=" * 100)
    
    print("\n  Key takeaways:")
    print("  • Daily resolution: Captures weekly patterns, but high noise reduces R²")
    print("  • Weekly resolution: Removes day-of-week noise, reveals monthly/quarterly patterns")
    print("  • Monthly resolution: Cleanest view of yearly seasonality, but loses short-term cycles")
    print("\n  Recommendation: Use weekly aggregation for forecasting models to balance")
    print("  signal clarity with temporal granularity.")


def print_fft_summary(fft_results: Dict, ts: pd.Series):
    """Print summary statistics of FFT analysis."""
    print("\n" + "=" * 80)
    print("FFT ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Time series length: {len(ts):,} days")
    print(f"Date range: {ts.index.min()} to {ts.index.max()}")
    print(f"Mean daily volume: {ts.mean():.2f}")
    print(f"Std daily volume: {ts.std():.2f}")
    
    print("\n" + "-" * 80)
    print("TOP 10 DOMINANT FREQUENCIES")
    print("-" * 80)
    
    top_freqs = identify_known_periods(fft_results['dominant_freqs'])
    display_df = top_freqs.head(10)[['period_days', 'frequency', 'power', 'amplitude', 'period_name']].copy()
    display_df['period_days'] = display_df['period_days'].round(2)
    display_df['power'] = display_df['power'].round(2)
    display_df['amplitude'] = display_df['amplitude'].round(2)
    display_df = display_df.fillna('')
    
    print(display_df.to_string(index=False))
    
    # Check for known periods
    print("\n" + "-" * 80)
    print("IDENTIFIED KNOWN PERIODS")
    print("-" * 80)
    known = top_freqs[top_freqs['period_name'].notna()]
    if len(known) > 0:
        for _, row in known.iterrows():
            print(f"  {row['period_name']}: Period = {row['period_days']:.2f} days, "
                  f"Power = {row['power']:.2f}, Amplitude = {row['amplitude']:.2f}")
    else:
        print("  No known periods clearly identified in top frequencies.")
    
    # Reconstruction quality
    print("\n" + "-" * 80)
    print("RECONSTRUCTION QUALITY (Top 10 Components)")
    print("-" * 80)
    reconstructed = reconstruct_signal(fft_results, n_components=10, original_mean=ts.values.mean())
    mse = np.mean((ts.values - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ts.values - reconstructed))
    r2 = 1 - (np.sum((ts.values - reconstructed) ** 2) / 
              np.sum((ts.values - ts.values.mean()) ** 2))
    
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Explained variance: {r2*100:.2f}%")


def run_multiresolution_analysis(df: pd.DataFrame, 
                                  output_dir: Path,
                                  sites: List[Optional[str]] = [None]):
    """
    Run multi-resolution FFT analysis for specified sites.
    
    Args:
        df: Raw dataframe
        output_dir: Directory to save visualizations
        sites: List of sites to analyze (None = all sites combined)
    """
    print("\n" + "=" * 100)
    print("MULTI-RESOLUTION FFT ANALYSIS")
    print("=" * 100)
    
    resolutions = ['D', 'W', 'ME']  # Daily, Weekly, Monthly
    
    for site in sites:
        site_name = f'Site {site}' if site else 'All Sites Combined'
        print(f"\n{'-' * 100}")
        print(f"Analyzing: {site_name}")
        print(f"{'-' * 100}")
        
        # Perform multi-resolution analysis
        results = perform_multiresolution_fft(df, site=site, resolutions=resolutions)
        
        if not results:
            print(f"  No results for {site_name}")
            continue
        
        # Print summary
        print_multiresolution_summary(results)
        
        # Create visualization
        save_name = site_name.lower().replace(' ', '_')
        save_path = output_dir / f'multiresolution_fft_{save_name}.png'
        plot_multiresolution_fft(results, site=site, save_path=save_path)
        
        print(f"\n  Multi-resolution analysis complete for {site_name}")


def main():
    """Main EDA function for FFT analysis."""
    print("=" * 80)
    print("FFT TIME SERIES DECOMPOSITION - ED VOLUME ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\nLoading dataset...")
    df = load_dataset()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'Visualizations' / 'FFT'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # MULTI-RESOLUTION FFT ANALYSIS (NEW)
    # =========================================================================
    run_multiresolution_analysis(df, output_dir, sites=[None, 'A', 'B', 'C', 'D'])
    
    # =========================================================================
    # STANDARD SINGLE-RESOLUTION ANALYSIS (ORIGINAL)
    # =========================================================================
    print("\n" + "=" * 100)
    print("STANDARD DAILY FFT ANALYSIS")
    print("=" * 100)
    
    # Analysis options
    analyses = [
        {'name': 'All Sites Combined', 'site': None, 'block': None},
        {'name': 'Site A', 'site': 'A', 'block': None},
        {'name': 'Site B', 'site': 'B', 'block': None},
        {'name': 'Site C', 'site': 'C', 'block': None},
        {'name': 'Site D', 'site': 'D', 'block': None},
    ]
    
    # Add block-specific analyses for all sites
    for block in range(4):
        analyses.append({
            'name': f'All Sites - Block {block}',
            'site': None,
            'block': block
        })
    
    # Run analyses
    for analysis in analyses:
        print("\n" + "=" * 80)
        print(f"ANALYSIS: {analysis['name']}")
        print("=" * 80)
        
        # Aggregate time series
        if analysis['block'] is not None:
            ts = aggregate_to_block_daily(df, site=analysis['site'], block=analysis['block'])
        else:
            ts = aggregate_to_daily(df, site=analysis['site'], block=analysis['block'])
        
        if len(ts) < 100:
            print(f"  Skipping: insufficient data ({len(ts)} days)")
            continue
        
        # Perform FFT
        fft_results = perform_fft_analysis(ts, sampling_rate=1.0, detrend=True)
        
        # Print summary
        print_fft_summary(fft_results, ts)
        
        # Create visualization
        save_name = analysis['name'].lower().replace(' ', '_').replace('-', '_')
        save_path = output_dir / f'fft_analysis_{save_name}.png'
        plot_fft_analysis(ts, fft_results, 
                         site=analysis['site'], 
                         block=analysis['block'],
                         save_path=save_path)
        
        print(f"\n  Analysis complete. Visualization saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("FFT ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {output_dir}")


def main_multiresolution_only():
    """Run only multi-resolution FFT analysis (faster for testing)."""
    print("=" * 100)
    print("MULTI-RESOLUTION FFT ANALYSIS ONLY")
    print("=" * 100)
    
    # Load data
    print("\nLoading dataset...")
    df = load_dataset()
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'Visualizations' / 'FFT'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run multi-resolution analysis for all sites combined
    run_multiresolution_analysis(df, output_dir, sites=[None])
    
    print("\n" + "=" * 100)
    print("MULTI-RESOLUTION FFT ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nVisualization saved to: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FFT Time Series Analysis')
    parser.add_argument('--multiresolution-only', '-m', action='store_true',
                       help='Run only multi-resolution FFT analysis (faster)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run full analysis including all sites and blocks')
    args = parser.parse_args()
    
    if args.multiresolution_only:
        main_multiresolution_only()
    else:
        main()
