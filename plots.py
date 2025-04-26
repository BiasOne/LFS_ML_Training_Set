import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The cost_function is kept exactly as provided
# We'll now create functions that use your cost_function to generate plots

def plot_storage_costs_comparison():
    """
    Plot comparing base storage costs between hot and cold based on file size
    """
    sizes = np.linspace(10, 1000, 20)  # 10MB to 1000MB
    days = 90  # 3 months
    
    hot_storage_costs = []
    cold_storage_costs = []
    
    for size in sizes:
        # We're only interested in storage costs, so set views to 0
        result = cost_function(days, 0, 0, 0, size, -1)
        hot_storage_costs.append(result["static_hot_storage_cost"])
        cold_storage_costs.append(result["static_cold_storage_cost"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, hot_storage_costs, label='Hot Storage', color='red', marker='o')
    plt.plot(sizes, cold_storage_costs, label='Cold Storage', color='blue', marker='s')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Storage Cost ($)')
    plt.title('Base Storage Cost Comparison: Hot vs Cold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('storage_cost_comparison.png')
    plt.close()

def plot_access_costs_comparison():
    """
    Plot comparing access costs between hot and cold based on number of views
    """
    views_range = np.logspace(1, 4, 20)  # 10 to 10,000 views (log scale)
    size_mb = 100  # 100MB file
    days = 90
    
    hot_access_costs = []
    cold_access_costs = []
    
    for views in views_range:
        # Get costs for hot storage
        hot_result = cost_function(days, 0, views, 0, size_mb, -1)
        hot_access = hot_result["static_hot_get_cost"] + hot_result["static_hot_retrieval_cost"] + hot_result["static_hot_network_cost"]
        hot_access_costs.append(hot_access)
        
        # Get costs for cold storage
        cold_result = cost_function(0, days, 0, views, size_mb, -1)
        cold_access = cold_result["static_cold_get_cost"] + cold_result["static_cold_retrieval_cost"] + cold_result["static_cold_network_cost"]
        cold_access_costs.append(cold_access)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(views_range, hot_access_costs, label='Hot Storage', color='red', marker='o')
    plt.semilogx(views_range, cold_access_costs, label='Cold Storage', color='blue', marker='s')
    plt.xlabel('Number of Views (log scale)')
    plt.ylabel('Access Cost ($)')
    plt.title('Access Cost Comparison: Hot vs Cold (100MB File)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('access_cost_comparison.png')
    plt.close()

def plot_total_costs_comparison():
    """
    Plot comparing total costs between hot, cold, and dynamic tiered storage
    """
    views_range = np.logspace(1, 4, 20)  # 10 to 10,000 views (log scale)
    size_mb = 100
    days = 90
    promotion_day = 30  # Promote to hot after 30 days
    
    hot_total_costs = []
    cold_total_costs = []
    tiered_total_costs = []
    
    for views in views_range:
        # For simplicity, distribute views proportionally between before/after promotion
        views_cold = views * promotion_day / days
        views_hot = views * (days - promotion_day) / days
        
        # Get costs for each strategy
        hot_result = cost_function(days, 0, views, 0, size_mb, -1)
        cold_result = cost_function(0, days, 0, views, size_mb, -1)
        tiered_result = cost_function(days - promotion_day, promotion_day, views_hot, views_cold, size_mb, promotion_day)
        
        hot_total_costs.append(hot_result["total_cost_hot_static"])
        cold_total_costs.append(cold_result["total_cost_cold_static"])
        tiered_total_costs.append(tiered_result["total_cost_with_tiering"])
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(views_range, hot_total_costs, label='Hot Storage', color='red', marker='o')
    plt.semilogx(views_range, cold_total_costs, label='Cold Storage', color='blue', marker='s')
    plt.semilogx(views_range, tiered_total_costs, label='Dynamic Tiered', color='green', marker='^')
    plt.xlabel('Number of Views (log scale)')
    plt.ylabel('Total Cost ($)')
    plt.title('Total Cost Comparison: Hot vs Cold vs Tiered (100MB File)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('total_cost_comparison.png')
    plt.close()

def plot_promotion_day_impact():
    """
    Plot comparing total costs for different promotion days
    """
    promotion_days = np.arange(0, 90, 5)  # Promotion days from 0 to 85
    views = 1000
    size_mb = 100
    days = 90
    
    tiered_costs = []
    hot_result = cost_function(days, 0, views, 0, size_mb, -1)
    cold_result = cost_function(0, days, 0, views, size_mb, -1)
    hot_cost = hot_result["total_cost_hot_static"]
    cold_cost = cold_result["total_cost_cold_static"]
    
    for promotion_day in promotion_days:
        # Calculate views for each phase
        views_cold = views * promotion_day / days
        views_hot = views * (days - promotion_day) / days
        
        tiered_result = cost_function(days - promotion_day, promotion_day, views_hot, views_cold, size_mb, promotion_day)
        tiered_costs.append(tiered_result["total_cost_with_tiering"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(promotion_days, tiered_costs, label='Dynamic Tiered', color='green', marker='o')
    plt.axhline(y=hot_cost, color='red', linestyle='--', label='Hot Storage')
    plt.axhline(y=cold_cost, color='blue', linestyle='--', label='Cold Storage')
    plt.xlabel('Promotion Day (Cold to Hot)')
    plt.ylabel('Total Cost ($)')
    plt.title('Impact of Promotion Day on Total Cost (100MB File, 1000 Views)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('promotion_day_impact.png')
    plt.close()

def plot_file_size_impact():
    """
    Plot showing how file size affects optimal strategy
    """
    sizes = np.logspace(0, 3, 20)  # 1MB to 1000MB (log scale)
    views = 500
    days = 90
    promotion_day = 30
    
    hot_costs = []
    cold_costs = []
    tiered_costs = []
    
    for size in sizes:
        views_cold = views * promotion_day / days
        views_hot = views * (days - promotion_day) / days
        
        hot_result = cost_function(days, 0, views, 0, size, -1)
        cold_result = cost_function(0, days, 0, views, size, -1)
        tiered_result = cost_function(days - promotion_day, promotion_day, views_hot, views_cold, size, promotion_day)
        
        hot_costs.append(hot_result["total_cost_hot_static"])
        cold_costs.append(cold_result["total_cost_cold_static"])
        tiered_costs.append(tiered_result["total_cost_with_tiering"])
    
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, hot_costs, label='Hot Storage', color='red', marker='o')
    plt.loglog(sizes, cold_costs, label='Cold Storage', color='blue', marker='s')
    plt.loglog(sizes, tiered_costs, label='Dynamic Tiered', color='green', marker='^')
    plt.xlabel('File Size (MB) - Log Scale')
    plt.ylabel('Total Cost ($) - Log Scale')
    plt.title('Impact of File Size on Total Cost (500 Views)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('file_size_impact.png')
    plt.close()

def plot_cost_components():
    """
    Plot showing the breakdown of cost components for each strategy
    """
    size_mb = 100
    views = 1000
    days = 90
    promotion_day = 30
    
    views_cold = views * promotion_day / days
    views_hot = views * (days - promotion_day) / days
    
    hot_result = cost_function(days, 0, views, 0, size_mb, -1)
    cold_result = cost_function(0, days, 0, views, size_mb, -1)
    tiered_result = cost_function(days - promotion_day, promotion_day, views_hot, views_cold, size_mb, promotion_day)
    
    # Extract cost components
    strategies = ['Hot Storage', 'Cold Storage', 'Dynamic Tiered']
    
    storage_costs = [
        hot_result["static_hot_storage_cost"],
        cold_result["static_cold_storage_cost"],
        tiered_result["storage_cost_hot"] + tiered_result["storage_cost_cold"]
    ]
    
    access_costs = [
        hot_result["static_hot_get_cost"] + hot_result["static_hot_retrieval_cost"] + hot_result["static_hot_network_cost"],
        cold_result["static_cold_get_cost"] + cold_result["static_cold_retrieval_cost"] + cold_result["static_cold_network_cost"],
        tiered_result["access_cost_hot"] + tiered_result["access_cost_cold"]
    ]
    
    transition_costs = [0, 0, tiered_result["transition_cost"]]
    
    # Create the stacked bar chart
    plt.figure(figsize=(12, 8))
    
    width = 0.6
    bottom = np.zeros(3)
    
    p1 = plt.bar(strategies, storage_costs, width, label='Storage Cost', bottom=bottom)
    bottom += storage_costs
    
    p2 = plt.bar(strategies, access_costs, width, label='Access Cost', bottom=bottom)
    bottom += access_costs
    
    p3 = plt.bar(strategies, transition_costs, width, label='Transition Cost', bottom=bottom)
    
    plt.ylabel('Cost ($)')
    plt.title('Cost Components by Storage Strategy (100MB File, 1000 Views)')
    plt.legend()
    
    # Add cost values as text labels
    for i, strategy in enumerate(strategies):
        plt.text(i, storage_costs[i]/2, f'${storage_costs[i]:.2f}', ha='center', va='center', color='white')
        plt.text(i, storage_costs[i] + access_costs[i]/2, f'${access_costs[i]:.2f}', ha='center', va='center', color='white')
        if transition_costs[i] > 0:
            plt.text(i, storage_costs[i] + access_costs[i] + transition_costs[i]/2, f'${transition_costs[i]:.2f}', ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig('cost_components.png')
    plt.close()