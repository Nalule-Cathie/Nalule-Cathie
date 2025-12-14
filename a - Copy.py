import pandas as pd
import numpy as np
from itertools import combinations, chain
from collections import defaultdict

# ======================
# PART A: DATA PREPARATION
# ======================

print("=" * 50)
print("PART A: DATA PREPARATION")
print("=" * 50)

# 1. Load the dataset into a dataframe
data = {
    'Transaction_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Items': [
        ['Bread', 'Milk', 'Eggs'],
        ['Bread', 'Butter'],
        ['Milk', 'Diapers', 'Beer'],
        ['Bread', 'Milk', 'Butter'],
        ['Milk', 'Diapers', 'Bread'],
        ['Beer', 'Diapers'],
        ['Bread', 'Milk', 'Eggs', 'Butter'],
        ['Eggs', 'Milk'],
        ['Bread', 'Diapers', 'Beer'],
        ['Milk', 'Butter']
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("1. Original Dataset:")
print(df.to_string(index=False))
print("\n")

# Convert to transaction format
transactions = df['Items'].tolist()
print("Transaction format:")
for i, transaction in enumerate(transactions, 1):
    print(f"T{i}: {transaction}")
print("\n")

# 2. One-hot encode the transaction data
def one_hot_encode(transactions):
    """Convert transactions to one-hot encoded format"""
    # Get all unique items
    all_items = sorted(set(chain(*transactions)))
    item_to_index = {item: idx for idx, item in enumerate(all_items)}
    
    # Create one-hot encoded matrix
    encoded_matrix = np.zeros((len(transactions), len(all_items)), dtype=int)
    
    for i, transaction in enumerate(transactions):
        for item in transaction:
            encoded_matrix[i, item_to_index[item]] = 1
    
    # Create DataFrame for better visualization
    encoded_df = pd.DataFrame(encoded_matrix, columns=all_items)
    encoded_df.index = [f'T{i+1}' for i in range(len(transactions))]
    
    return encoded_df, item_to_index

encoded_df, item_index = one_hot_encode(transactions)
print("2. One-hot Encoded Transactions:")
print(encoded_df)
print("\n")

# ======================
# PART B: APRIORI ALGORITHM
# ======================

print("=" * 50)
print("PART B: APRIORI ALGORITHM")
print("=" * 50)

class Apriori:
    def __init__(self, transactions, min_support=0.2, min_confidence=0.5):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        
    def get_support(self, itemset):
        """Calculate support for an itemset"""
        count = 0
        for transaction in self.transactions:
            if all(item in transaction for item in itemset):
                count += 1
        return count / len(self.transactions)
    
    def generate_candidates(self, itemsets, k):
        """Generate candidate itemsets of size k"""
        candidates = set()
        itemsets = list(itemsets)
        
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Join two itemsets if they have k-2 items in common
                union_set = set(itemsets[i]).union(set(itemsets[j]))
                if len(union_set) == k:
                    # Check if all subsets of size k-1 are frequent
                    subsets = list(combinations(union_set, k-1))
                    if all(subset in itemsets for subset in subsets):
                        candidates.add(tuple(sorted(union_set)))
        
        return [list(candidate) for candidate in candidates]
    
    def find_frequent_itemsets(self):
        """Find all frequent itemsets"""
        # Get all unique items
        all_items = sorted(set(chain(*self.transactions)))
        
        # Generate frequent 1-itemsets
        frequent_1 = []
        for item in all_items:
            support = self.get_support([item])
            if support >= self.min_support:
                frequent_1.append([item])
        
        self.frequent_itemsets[1] = frequent_1
        
        # Generate frequent k-itemsets
        k = 2
        while True:
            candidates = self.generate_candidates(self.frequent_itemsets[k-1], k)
            frequent_k = []
            
            for candidate in candidates:
                support = self.get_support(candidate)
                if support >= self.min_support:
                    frequent_k.append(candidate)
            
            if not frequent_k:
                break
            
            self.frequent_itemsets[k] = frequent_k
            k += 1
    
    def generate_rules(self):
        """Generate association rules from frequent itemsets"""
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:  # Need at least 2 items for rules
                continue
            
            for itemset in itemsets:
                itemset_support = self.get_support(itemset)
                
                # Generate all non-empty proper subsets
                for i in range(1, k):
                    for antecedent in combinations(itemset, i):
                        antecedent = list(antecedent)
                        consequent = [item for item in itemset if item not in antecedent]
                        
                        antecedent_support = self.get_support(antecedent)
                        confidence = itemset_support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            lift = confidence / self.get_support(consequent)
                            self.rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': round(itemset_support, 3),
                                'confidence': round(confidence, 3),
                                'lift': round(lift, 3)
                            })
    
    def run(self):
        """Run the complete Apriori algorithm"""
        print(f"Parameters: Minimum Support = {self.min_support}, Minimum Confidence = {self.min_confidence}")
        print("-" * 50)
        
        # Find frequent itemsets
        print("\n1. Finding Frequent Itemsets...")
        self.find_frequent_itemsets()
        
        print("\nFrequent Itemsets Found:")
        for k, itemsets in self.frequent_itemsets.items():
            print(f"  {k}-itemsets ({len(itemsets)}):")
            for itemset in itemsets:
                support = self.get_support(itemset)
                print(f"    {itemset} (support: {support:.3f})")
        
        # Generate rules
        print("\n2. Generating Association Rules...")
        self.generate_rules()
        
        # Display rules
        print(f"\nGenerated {len(self.rules)} association rules:")
        print("-" * 80)
        print(f"{'Rule':<30} {'Support':<10} {'Confidence':<12} {'Lift':<10}")
        print("-" * 80)
        
        for rule in self.rules:
            rule_str = f"{rule['antecedent']} => {rule['consequent']}"
            print(f"{rule_str:<30} {rule['support']:<10} {rule['confidence']:<12} {rule['lift']:<10}")
        
        return self.rules

# Run Apriori algorithm
print("Running Apriori Algorithm...")
apriori = Apriori(transactions, min_support=0.2, min_confidence=0.5)
rules = apriori.run()

# PART C: INTERPRETATION


print("\n" + "=" * 50)
print("PART C: INTERPRETATION")
print("=" * 50)

# 1. Identify three strongest rules based on lift
print("\n1. THREE STRONGEST RULES (by Lift):")
print("-" * 60)

# Sort rules by lift in descending order
sorted_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)

for i, rule in enumerate(sorted_rules[:3], 1):
    print(f"\nRule {i}: {rule['antecedent']} => {rule['consequent']}")
    print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
    
    # Interpretation
    antecedent_str = " & ".join(rule['antecedent'])
    consequent_str = " & ".join(rule['consequent'])
    
    if rule['lift'] > 1:
        print(f"  Interpretation: Customers who buy {antecedent_str} are {rule['lift']:.1f}x MORE likely to also buy {consequent_str}")
    elif rule['lift'] < 1:
        print(f"  Interpretation: Customers who buy {antecedent_str} are {1/rule['lift']:.1f}x LESS likely to also buy {consequent_str}")
    else:
        print(f"  Interpretation: Purchases of {antecedent_str} and {consequent_str} are independent")

# 2. Business recommendations
print("\n" + "-" * 60)
print("2. BUSINESS RECOMMENDATIONS")
print("-" * 60)

print("\nRecommendation 1: BUNDLE PROMOTIONS")
print("• Based on the strong association between Milk and Bread (lift > 1):")
print("  - Create 'Breakfast Essentials' bundle: Milk + Bread + Eggs")
print("  - Offer 10% discount when purchased together")
print("  - Place these items in adjacent shelves or same aisle")

print("\nRecommendation 2: CROSS-SELLING STRATEGY")
print("• Based on the Beer-Diapers association:")
print("  - Create targeted promotions: 'Baby care + Relaxation' section")
print("  - Suggest Beer to customers buying Diapers (online recommendation)")
print("  - Consider demographic targeting for young parents")

print("\nRecommendation 3: INVENTORY MANAGEMENT")
print("• Based on frequent itemset patterns:")
print("  - Stock Milk and Bread together near store entrance")
print("  - Ensure Butter is available near both Milk and Bread sections")
print("  - Monitor Diapers-Beer purchases for weekend stock planning")

print("\nRecommendation 4: LOYALTY PROGRAM")
print("• Based on multi-item purchase patterns:")
print("  - Reward customers who buy 3+ frequently associated items")
print("  - Offer 'Complete Your Meal' suggestions at checkout")
print("  - Create personalized coupons based on purchase history")

# ======================
# ADDITIONAL ANALYSIS
# ======================

print("\n" + "=" * 50)
print("ADDITIONAL ANALYSIS")
print("=" * 50)

# Summary statistics
print("\nDataset Summary:")
print(f"• Total transactions: {len(transactions)}")
print(f"• Total unique items: {len(set(chain(*transactions)))}")
print(f"• Average items per transaction: {np.mean([len(t) for t in transactions]):.2f}")

# Most frequent items
item_counts = defaultdict(int)
for transaction in transactions:
    for item in transaction:
        item_counts[item] += 1

print("\nItem Frequency:")
sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
for item, count in sorted_items:
    print(f"  {item}: {count} purchases ({count/len(transactions)*100:.0f}% of transactions)")

print("\n" + "=" * 50)
print("EXECUTION COMPLETE")
print("=" * 50)