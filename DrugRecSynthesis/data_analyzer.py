import pandas as pd
import matplotlib.pyplot as plt
import os
from args import arg
from pylab import mpl
# Configure matplotlib to support Chinese display
import matplotlib.font_manager as fm
import matplotlib

try:
    if hasattr(matplotlib.font_manager, '_rebuild'):
        matplotlib.font_manager._rebuild()
    else:
        matplotlib.font_manager.fontManager.__init__()
    print("Font cache rebuilt")
except Exception as e:
    print(f"Font cache rebuild failed, continuing: {e}")

# Find Chinese fonts
print("=== Finding Chinese fonts in the system ===")
chinese_fonts = []
for font in fm.fontManager.ttflist:
    if 'CJK' in font.name or 'Noto' in font.name or 'WenQuanYi' in font.name or 'SimHei' in font.name:
        chinese_fonts.append(font.name)

# Set font
if chinese_fonts:
    chosen_font = chinese_fonts[0]
    print(f"Found and using Chinese font: {chosen_font}")
    plt.rcParams['font.sans-serif'] = [chosen_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue
else:
    print("No Chinese fonts found, using default font")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

class DataAnalyzer:
    """Data analysis class for statistical analysis of generated population data"""
    
    def __init__(self, output_dir=None):
        """
        Initialize data analyzer
        
        Args:
            output_dir: Output directory path, uses default path if None
        """
        self.output_dir = output_dir or f"output/{arg.out_doc}"
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def group_analysis(self, people_data):
        """
        Statistical analysis by population groups, including:
        Basic groups: children, adolescents, adults, elderly
        Special groups: pregnant women, liver dysfunction, kidney dysfunction, etc.
        
        Args:
            people_data: Population data list
            
        Returns:
            dict: Group statistics results
        """
        # Collect all group information
        all_groups = []
        total_people = len(people_data)
        
        # Define all possible group categories (sorted by priority)
        group_categories = [
            '儿童', '青少年', '成人', '老年人', 
            '孕妇', '哺乳期', 
            '肝功能不全', '肾功能不全'
        ]
        
        # Count occurrences of each group
        group_counts = {}
        
        for person in people_data:
            person_groups = person['group'] if isinstance(person['group'], list) else [person['group']]
            
            # Add each person's groups to the total list
            for group in person_groups:
                if group in group_counts:
                    group_counts[group] += 1
                else:
                    group_counts[group] = 1
                all_groups.append(group)
        
        # Ensure all predefined groups are in statistics (even if count is 0)
        for category in group_categories:
            if category not in group_counts:
                group_counts[category] = 0
        
        # Sort by predefined order
        ordered_groups = {}
        for category in group_categories:
            if category in group_counts:
                ordered_groups[category] = group_counts[category]
        
        # Add other undefined groups
        for group, count in group_counts.items():
            if group not in ordered_groups:
                ordered_groups[group] = count
        
        # Calculate percentages and output statistics
        self._print_group_statistics(ordered_groups, total_people)
        
        # Draw bar chart
        self._plot_group_distribution(ordered_groups)
        
        return ordered_groups
    
    def age_analysis(self, people_data):
        """
        Age distribution statistical analysis
        Count population distribution by age intervals
        
        Args:
            people_data: Population data list
            
        Returns:
            dict: Age analysis results
        """
        # Collect age data
        ages = []
        total_people = len(people_data)
        
        for person in people_data:
            # Support both dictionary and object formats
            age = person['age'] if isinstance(person, dict) else person.age
            ages.append(age)
        
        # Define age intervals (left-closed, right-open)
        age_ranges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        age_labels = ["0-9岁", "10-19岁", "20-29岁", "30-39岁", "40-49岁", 
                      "50-59岁", "60-69岁", "70-79岁", "80-89岁", "90-99岁"]
        
        # Use pandas for age interval classification
        age_series = pd.Series(ages)
        age_bins = pd.cut(age_series, bins=age_ranges, right=False, include_lowest=True, labels=age_labels)
        
        # Calculate number of people in each age interval
        age_distribution = age_bins.value_counts().sort_index()
        
        # Output statistics
        self._print_age_statistics(ages, age_labels, age_distribution, total_people)

        # Draw bar chart
        self._plot_age_distribution(age_labels, age_distribution)
        
        return {
            'age_distribution': dict(zip(age_labels, age_distribution)),
            'stats': {
                'total': total_people,
                'min_age': min(ages),
                'max_age': max(ages),
                'avg_age': sum(ages)/len(ages)
            }
        }

    def gender_analysis(self, people_data):
        """
        Gender distribution statistical analysis
        Count distribution of males and females
        
        Args:
            people_data: Population data list
            
        Returns:
            dict: Gender analysis results
        """
        # Collect gender data
        gender_list = []
        total_people = len(people_data)
        
        for person in people_data:
            gender_list.append(person['gender'])
        
        # Calculate male and female counts
        gender_series = pd.Series(gender_list)
        gender_counts = gender_series.value_counts()
        
        # Output statistics
        self._print_gender_statistics(gender_counts, total_people)
        
        # Draw bar chart
        self._plot_gender_distribution(gender_counts)
        
        return {
            'gender_distribution': dict(gender_counts),
            'stats': {
                'total': total_people,
                'male_count': gender_counts.get('男', 0),
                'female_count': gender_counts.get('女', 0)
            }
        }
    
    def _print_group_statistics(self, ordered_groups, total_people):
        """Print population group statistics"""
        print("=" * 50)
        print("Generated Population Group Statistics:")
        print("=" * 50)
        print(f"Total population: {total_people}")
        print("-" * 30)
        
        # Basic age group statistics
        print("Basic age groups:")
        age_groups = ['儿童', '青少年', '成人', '老年人']
        for group in age_groups:
            count = ordered_groups.get(group, 0)
            percentage = (count / total_people) * 100 if total_people > 0 else 0
            print(f"  {group}: {count} people ({percentage:.2f}%)")
        
        print("-" * 30)
        
        # Special population statistics
        print("Special population groups:")
        special_groups = ['孕妇', '哺乳期', '肝功能不全', '肾功能不全']
        for group in special_groups:
            count = ordered_groups.get(group, 0)
            percentage = (count / total_people) * 100 if total_people > 0 else 0
            if count > 0:  # Only show special groups with population
                print(f"  {group}: {count} people ({percentage:.2f}%)")
        
        # Other groups
        other_groups = {k: v for k, v in ordered_groups.items() 
                       if k not in age_groups and k not in special_groups and v > 0}
        if other_groups:
            print("-" * 30)
            print("Other groups:")
            for group, count in other_groups.items():
                percentage = (count / total_people) * 100 if total_people > 0 else 0
                print(f"  {group}: {count} people ({percentage:.2f}%)")
        
        print("=" * 50)
    
    def _print_age_statistics(self, ages, age_labels, age_distribution, total_people):
        """Print age statistics"""
        # Calculate percentages
        age_distribution_percent = (age_distribution / total_people) * 100
        
        print("=" * 50)
        print("Age Distribution Statistics:")
        print("=" * 50)
        print(f"Total population: {total_people}")
        print(f"Age range: {min(ages)} years old - {max(ages)} years old")
        print(f"Average age: {sum(ages)/len(ages):.1f} years old")
        print("-" * 30)
        
        print("Age interval groups:")
        for label, count, percent in zip(age_labels, age_distribution, age_distribution_percent):
            print(f"  {label}: {count} people ({percent:.2f}%)")
        print("=" * 50)
    
    def _print_gender_statistics(self, gender_counts, total_people):
        """Print gender statistics"""
        print("=" * 50)
        print("Gender Distribution Statistics:")
        print("=" * 50)
        print(f"Total population: {total_people}")
        print("-" * 30)
        
        # Calculate percentage for each gender and output
        print("Gender groups:")
        for gender in ['男', '女']:
            count = gender_counts.get(gender, 0)
            percentage = (count / total_people) * 100 if total_people > 0 else 0
            print(f"  {gender}: {count} people ({percentage:.2f}%)")
        
        print("=" * 50)
    
    def _plot_group_distribution(self, ordered_groups):
        """Draw population group distribution chart"""
        plt.figure(figsize=(14, 8))
        
        # Prepare plot data (only show groups with population)
        plot_groups = {k: v for k, v in ordered_groups.items() if v > 0}
        
        if plot_groups:
            groups_names = list(plot_groups.keys())
            groups_counts = list(plot_groups.values())
            
            # Create bar chart
            bars = plt.bar(groups_names, groups_counts, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'][:len(groups_names)])
            
            # Display count on each bar
            for bar, count in zip(bars, groups_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(groups_counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('Population Groups', fontsize=12, fontweight='bold')
            plt.ylabel('Population Count', fontsize=12, fontweight='bold')
            plt.title('Population Group Distribution Statistics', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            
            # Adjust layout to avoid label cutoff
            plt.tight_layout()
            
            # Add grid lines
            plt.grid(axis='y', alpha=0.3)
            
            # Save image
            plt.savefig(f"{self.output_dir}/group_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Warning: No valid group data for plotting")
    
    def _plot_age_distribution(self, age_labels, age_distribution):
        """Draw age distribution chart"""
        plt.figure(figsize=(14, 8))
        
        # Prepare plot data (remove zero value intervals)
        plot_labels = []
        plot_counts = []
        for label, count in zip(age_labels, age_distribution):
            if count > 0:
                plot_labels.append(label)
                plot_counts.append(count)
        
        if plot_counts:
            # Create bar chart - use same color scheme as group_analysis
            bars = plt.bar(plot_labels, plot_counts, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                                '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', 
                                '#FF9F43', '#54A0FF'][:len(plot_labels)])
            
            # Display count on each bar
            for bar, count in zip(bars, plot_counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(plot_counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('Age Intervals', fontsize=12, fontweight='bold')
            plt.ylabel('Population Count', fontsize=12, fontweight='bold')
            plt.title('Age Distribution Statistics', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            
            # Adjust layout to avoid label cutoff
            plt.tight_layout()
            
            # Add grid lines
            plt.grid(axis='y', alpha=0.3)
            
            # Save image
            plt.savefig(f"{self.output_dir}/age_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Warning: No valid age data for plotting")
    
    def _plot_gender_distribution(self, gender_counts):
        """Draw gender distribution chart"""
        plt.figure(figsize=(14, 8))
        
        # Prepare plot data
        if len(gender_counts) > 0:
            genders = list(gender_counts.index)
            counts = list(gender_counts.values)
            
            # Create bar chart - use same color scheme
            bars = plt.bar(genders, counts, 
                          color=['#FF6B6B', '#4ECDC4'][:len(genders)])
            
            # Display count on each bar
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.xlabel('Gender', fontsize=12, fontweight='bold')
            plt.ylabel('Population Count', fontsize=12, fontweight='bold')
            plt.title('Gender Distribution Statistics', fontsize=14, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            
            # Add grid lines
            plt.grid(axis='y', alpha=0.3)
            
            # Save image
            plt.savefig(f"{self.output_dir}/gender_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Warning: No valid gender data for plotting") 