import argparse
parser = argparse.ArgumentParser()
# Probability of liver dysfunction
parser.add_argument("--liver_prob", type=float, default=0.04)
# Probability of kidney dysfunction
parser.add_argument("--kidney_prob", type=float, default=0.1)
# Whether to generate based on historical data (0: regenerate, 1: continue from historical data)
parser.add_argument("--history_data", type=int, default=0)
# Historical data folder
parser.add_argument("--history_doc", type=str, default="DrugRec_0724")
# Probability of having medical history
parser.add_argument("--medhistory_prob", type=float, default=1)
# Probability of having allergens
parser.add_argument("--allergen_prob", type=float, default=0.2)
# Number of people to generate
parser.add_argument("--people_num", type=int, default=5)
# Generated data save folder
parser.add_argument("--out_doc", type=str, default="DrugRec_0724")
# Source CSV file for allergen list extraction
parser.add_argument("--allergen_filename", type=str, default='data/allergen.csv')
# Source CSV file for symptom list extraction
parser.add_argument("--diagnosis_filename", type=str, default='data/diagnosis_medicine_dict.json')
# Population age probability file
parser.add_argument("--geography_file_path", type=str, default='data/agedemo2.csv')
# Medical history candidate list file
parser.add_argument("--medhistory_file_path", type=str, default='data/medical_history.csv')
# Probability of females being pregnant
parser.add_argument("--pregnant_prob", type=float, default=0.2)
# Probability of females being lactating
parser.add_argument("--lactation_prob", type=float, default=0.2)
# Whether to consider disease coverage
parser.add_argument("--consider_coverage", type=int, default=1)
# Maximum number of times the same disease can appear, only effective when considering disease coverage
parser.add_argument("--upper_limit", type=int, default=2)

arg = parser.parse_args()

if __name__ == "__main__":
    pass
