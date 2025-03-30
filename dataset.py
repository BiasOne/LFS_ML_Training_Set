import pandas as pd
import numpy as np

np.random.seed(42) # For reproducible results

# Kinds of files we might store
object_types = ['video', 'music', 'document', 'image']

def generate_access_pattern(trend_type, days):
    base = np.zeros(days)
    spike_day = None
    trend_start_day = None

    if trend_type == "stagnant":
        # Somewhat low random daily views
        base = np.random.randint(1, 10, size=days)
        for i in range(1, days):
            base[i] = max(0, base[i-1] + np.random.randint(-2, 3))

    elif trend_type == "gradual_increase":
        # Objects that slowly get more popular over time.
        max_value = np.random.randint(200, 500) # Random peak value
        base = np.linspace(1, max_value, days) # Linear increase
        # Add some relative noise
        noise_factor = 0.2 # up/down by up to 20% each day (TODO: maybe this should be random instead of a flag 20%)
        for i in range(days):
            current_val = base[i]
            noise = np.random.uniform(-noise_factor, noise_factor) * current_val
            base[i] = max(1, current_val + noise) # Ensure >= 1 view

        # Guess when the trend became significant (hits 25% of max)
        significant_increase_threshold = max_value * 0.25
        trend_start_day = 0
        for i in range(days):
            if base[i] >= significant_increase_threshold:
                spike_day = i # Mark when threshold is crossed
                # Assume trend started a bit earlier (2-5 days)
                trend_start_day = max(0, i - np.random.randint(2, 5))
                break

    elif trend_type == "viral_spike":
        # Simulate sudden popularity increase (viral content)
        base = np.random.randint(1, 20, size=days) # Start low

        spike_day = np.random.randint(days // 3, 2 * days // 3) # Spike happens mid-simulation

        # Simulate gradual buildup before the main spike
        buildup_days = np.random.randint(2, 14)
        spike_height = np.random.randint(500, 100000)

        trend_start_day = max(0, spike_day - buildup_days) # Trend starts with buildup

        # Pre-spike buildup
        for i in range(trend_start_day, spike_day):
            if i < days:
                progress = (i - trend_start_day) / max(1, buildup_days)
                # Increase views as a fraction of the peak
                base[i] = max(base[i], base[i] + (progress ** 2) * spike_height * np.random.uniform(0.1, 0.4))

        # Spike day and subsequent decay
        decay_rate = np.random.uniform(0.75, 0.95)

        if spike_day < days:
            base[spike_day] = max(base[spike_day], spike_height)

            # Decay after spike day, with some noise
            for i in range(spike_day + 1, days):
                decay_factor = decay_rate ** (i - spike_day)
                decay_noise = np.random.normal(1.0, 0.1) # Variable decay
                decayed_value = spike_height * decay_factor * decay_noise
                base[i] = max(np.random.randint(1, 20), decayed_value) # baseline view count
        elif spike_day == days: # Handle edge case where spike is last day
            base[days-1] = max(base[days-1], spike_height)

    # Simulate weekly seasonality (higher views on weekends)
    weekday_effect = np.array([1.0, 1.0, 1.0, 1.1, 1.2, 1.5, 1.3] * (days // 7 + 1))[:days]
    base = base * weekday_effect

    # Ensure integer views are >= 0
    base = np.maximum(0, np.round(base)).astype(int)

    return base, spike_day, trend_start_day

def calculate_pre_spike_features(access_pattern, spike_day):
    # Average views before/after a known spike day
    if spike_day == -1 or spike_day is None:
        return {"pre_spike_avg": -1, "post_spike_avg": -1} # Return -1 if no spike

    pre_spike_avg = np.mean(access_pattern[:spike_day]) if spike_day > 0 else 0.0
    post_spike_avg = np.mean(access_pattern[spike_day:]) if spike_day < len(access_pattern) else 0.0

    return {"pre_spike_avg": pre_spike_avg, "post_spike_avg": post_spike_avg}

def calculate_object_stats(access_pattern):
    # Calculate summary stats from the access pattern
    total_views = int(sum(access_pattern))

    mean = np.mean(access_pattern)
    std_dev = np.std(access_pattern)

    pattern_length = len(access_pattern)

    # Calculate averages over the first N days
    first_30d_avg = np.mean(access_pattern[0:min(pattern_length, 30)])
    first_14d_avg = np.mean(access_pattern[0:min(pattern_length, 14)])
    first_7d_avg = np.mean(access_pattern[0:min(pattern_length, 7)])
    first_3d_avg = np.mean(access_pattern[0:min(pattern_length, 3)])

    # Volatility: Coefficient of variation (std dev / mean)
    volatility = std_dev / max(mean, 1)

    # Trend Acceleration: Avg change in daily changes over last 5 days
    acceleration = 0.0 # Default
    if pattern_length >= 5:
        daily_changes = np.diff(access_pattern)
        # Ensure we have enough diffs to calculate the diff of diffs
        if len(daily_changes) >= 2:
             acceleration = np.mean(np.diff(daily_changes[-min(5, len(daily_changes)):]))

    return {
        "total_views": total_views,
        "avg_30d": first_30d_avg,
        "avg_14d": first_14d_avg,
        "avg_7d": first_7d_avg,
        "avg_3d": first_3d_avg,
        "volatility": volatility,
        "trend_acceleration": acceleration,
    }

# Simulate a social media trend score (higher for viral potential)
def generate_social_trend_score(trend_type, access_pattern, obj_type):
    base_score = 0

    # Base score depends on simulated trend type
    if trend_type == "viral_spike": base_score = np.random.randint(70, 100)
    elif trend_type == "gradual_increase": base_score = np.random.randint(30, 70)
    else: base_score = np.random.randint(0, 30) # stagnant

    # Boost score slightly for content types often shared socially
    if obj_type == "video": base_score += np.random.randint(0, 20)
    elif obj_type == "music": base_score += np.random.randint(0, 15)

    # Adjust based on recent activity (last 3 days vs overall)
    if len(access_pattern) >= 3:
        recency_factor = np.mean(access_pattern[-3:]) / max(np.mean(access_pattern), 1)
        base_score *= min(2, max(0.5, recency_factor)) # Clamp factor

    # Add random noise
    noise = np.random.uniform(-20, 20)
    base_score = base_score * (1 + noise/100)

    return min(100, max(0, base_score)) # Ensure score is 0-100

# Determine the optimal day to move object to hot storage based on simulated pattern
def determine_optimal_promotion_day(trend_type, spike_day, trend_start_day, access_pattern, obj_type, obj_size):
    # How latency sensitive is this content type? (higher = more sensitive)
    latency_sensitivity = {
        'video': 0.9, 'music': 0.8, 'document': 0.3, 'image': 0.4
    }

    base_threshold = 500 # Base daily views needed to consider promotion (TODO: Maybe this should depend on the obj type?)

    # Adjust threshold based on latency sensitivity (less sensitive types need more views)
    min_views_threshold = base_threshold / max(0.1, latency_sensitivity.get(obj_type, 0.6))

    # Adjust threshold by size (larger files need more views to justify cost/effort)
    if obj_size > 1000: min_views_threshold *= 1.5 # >1GB
    elif obj_size < 10: min_views_threshold *= 0.5 # <10MB

    # Determine promotion day based on trend type and thresholds

    if trend_type == "stagnant":
        # Promote only if views ever cross threshold
        if max(access_pattern) >= min_views_threshold:
            # Find first day threshold was crossed
            for day, views in enumerate(access_pattern):
                if views >= min_views_threshold: return day
        return -1 # dop not promote

    if trend_type == "viral_spike" and spike_day is not None:
        # Check if peak is high enough
        peak_views = 0
        if spike_day < len(access_pattern): peak_views = max(access_pattern[spike_day:])
        if peak_views < min_views_threshold: return -1 # Not worth it

        # Promote latency-sensitive types 1 day before spike
        if obj_type in ['video', 'music']: return max(0, spike_day - 1)
        else: # For others, decide based on sensitivity
             if latency_sensitivity.get(obj_type, 0.6) < 0.5: return spike_day # Promote on spike day if not sensitive
             else: return max(0, spike_day - 1) # Else promote day before

    if trend_type == "gradual_increase" and trend_start_day is not None:
        # Promote if trend eventually crosses threshold
        if max(access_pattern) < min_views_threshold: return -1

        # Find first day threshold is crossed *after* trend started
        for day in range(trend_start_day, len(access_pattern)):
            if access_pattern[day] >= min_views_threshold:
                # Promote sensitive types 1 day early, others on the day
                if latency_sensitivity.get(obj_type, 0.6) > 0.7: return max(0, day - 1)
                else: return day
        return -1 # Trend started but never crossed threshold after start_day

    return -1 # do not promote

# Estimate user experience impact based on latency sensitivity and views
def calculate_latency_impact(obj_type, views):
    latency_impact_factor = {
        'video': 0.9, 'music': 0.8, 'document': 0.3, 'image': 0.4
    }
    impact_score = views * latency_impact_factor.get(obj_type, 0.6) # Higher score = higher impact
    return impact_score

def main():
    data = [] # List to hold data rows

    num_samples = 1000
    days_to_simulate = 30

    for i in range(num_samples): # Main loop for generating samples
        object_id = f"obj_{i}"
        obj_type = np.random.choice(object_types)

        # Assign size based on object type (MB)
        size = 0
        if obj_type == 'video': size = np.random.randint(100, 5000)
        elif obj_type == 'music': size = np.random.randint(2, 7)
        elif obj_type == 'document': size = np.random.randint(1, 2)
        elif obj_type == 'image': size = np.random.randint(1, 4)

        object_age = np.random.randint(30, 181) # Simulate age 1-6 months

        # Adjust trend probability based on type (videos/music more likely viral)
        if obj_type in ["video", "music"]: trend_probs = [0.3, 0.4, 0.3] # stagnant, gradual, viral
        else: trend_probs = [0.6, 0.3, 0.1]

        trend_type = np.random.choice(["stagnant", "gradual_increase", "viral_spike"], p=trend_probs)

        # Generate the access pattern for this object
        access_counts, spike_day, trend_start_day = generate_access_pattern(trend_type, days_to_simulate)

        # Determine the target variable: optimal day for promotion
        optimal_promotion_day = determine_optimal_promotion_day(
            trend_type, spike_day, trend_start_day, access_counts, obj_type, size
        )

        # Simulate time since last access in seconds (TODO: Need to come up with something better because this doesnt really line up well with the rest of the data)
        seconds_per_day = 24*60*60
        time_since_last_access = 0
        if trend_type == "viral_spike" or trend_type == "gradual_increase":
             # Accessed more recently if trending
            if np.random.random() < 0.3: time_since_last_access = np.random.randint(0, 2 * 60 * 60) # Within 2 hours
            else: time_since_last_access = np.random.randint(2 * 60 * 60, seconds_per_day) # Within last day
        else: # Stagnant objects last accessed longer ago (1-30 days)
            time_since_last_access = np.random.randint(seconds_per_day, 30 * seconds_per_day)

        # Calculate summary statistics/features
        features = calculate_object_stats(access_counts) # Using renamed func

        # Calculate simulated social score
        social_trend_score = generate_social_trend_score(
            trend_type, access_counts, obj_type)

        # Derive days until promotion from optimal day
        days_until_optimal_promotion = optimal_promotion_day # -1 if never

        # Calculate pre/post spike averages
        spike_features = calculate_pre_spike_features(access_counts, spike_day)

        # Assemble the data row
        row_data = [
            object_id,
            features["total_views"],
            size,
            obj_type,
            object_age,
            time_since_last_access,
            social_trend_score,
            spike_day if spike_day is not None else -1,
            trend_start_day if trend_start_day is not None else -1,
            optimal_promotion_day,
            days_until_optimal_promotion,
            features["avg_30d"],
            features["avg_14d"],
            features["avg_7d"],
            features["avg_3d"],
            features["volatility"],
            features["trend_acceleration"],
            spike_features["pre_spike_avg"],
            spike_features["post_spike_avg"]
        ]

        # Add daily access counts to the end of the row
        row_data.extend(access_counts.tolist())

        data.append(row_data) # Add row to dataset list

    # Define column headers for the CSV
    columns = [
        "object_id",
        "total_views",
        "size_MB",
        "object_type",
        "object_age_days",
        "time_since_last_access",
        "social_trend_score",
        "spike_day",
        "trend_start_day",
        "optimal_promotion_day",
        "days_until_optimal_promotion",
        "avg_30d",
        "avg_14d",
        "avg_7d",
        "avg_3d",
        "volatility",
        "trend_acceleration",
        "pre_spike_avg",
        "post_spike_avg"
    ]

    # Add headers for daily view columns
    for day in range(days_to_simulate):
        columns.append(f"day_{day}_views")

    df = pd.DataFrame(data, columns=columns)

    file_path = "training_dataset.csv"
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    main()