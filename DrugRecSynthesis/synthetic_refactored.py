from py2neo import Graph
import json
import pickle
from LLMAPI import LLMAPI
from args import arg
import pandas as pd
import sys
import os
from tqdm import tqdm
import random
from DrugReview import DrugReviewSystem
from data_analyzer import DataAnalyzer


class Synthetic:
    """Synthetic data generation class"""
    
    def __init__(self, neo4j_uri="http://localhost:7474", 
                 api_key="", 
                 model="qwen-max"):
        """
        Initialize synthetic data generator
        
        Args:
            neo4j_uri: Neo4j database connection URI
            api_key: LLM API key
            model: Model name to use
        """
        self.graph = Graph(neo4j_uri)
        self.reviewer = DrugReviewSystem(neo4j_uri)
        self.llm_api = LLMAPI(api_key=api_key, model=model)
        self.llm_cache = {}
        self.data_analyzer = DataAnalyzer()
        
        # Data file loading status
        self._medicine_symptoms_dict = None
        self._age_probabilities = None
        self._allergen_list = None
        self._interaction_dict = None
        self._drugmsg_dict = None
    
    def _load_data_files(self):
        """Lazy loading of data files"""
        if self._medicine_symptoms_dict is None:
            with open(arg.diagnosis_filename, 'r', encoding='utf-8') as f:
                self._medicine_symptoms_dict = json.load(f)
        
        if self._age_probabilities is None:
            self._age_probabilities = pd.read_csv(arg.geography_file_path)
        
        if self._allergen_list is None:
            self._allergen_list = pd.read_csv(arg.allergen_filename)['allergen'].values.tolist()
        
        if self._interaction_dict is None:
            try:
                with open("data/drug_interaction_analysis_dict.pkl", "rb") as f:
                    self._interaction_dict = pickle.load(f)
            except FileNotFoundError:
                print("Warning: Drug interaction data file not found")
                self._interaction_dict = {}
        
        if self._drugmsg_dict is None:
            try:
                with open("data/drugMsg_linux_dict.pkl", "rb") as f:
                    self._drugmsg_dict = pickle.load(f)
            except FileNotFoundError:
                print("Warning: Drug information data file not found")
                self._drugmsg_dict = {}

    def check_diagnosis_reasonable(self, diagnosis, person):
        """
        Check if diagnosis is reasonable based on patient's gender and age characteristics
        
        Args:
            diagnosis: Diagnosis name
            person: Patient information containing age and gender fields
        
        Returns:
            bool: True if reasonable, False if not
        """
        # Define age-related keywords
        age_keywords = {
            'children': ['小儿', '儿童', '婴儿', '新生儿', '幼儿', '婴幼儿', '小孩', '先天性'],
            'adult': ['成人'],
            'elderly': ['老年', '老人', '退行性', '老年性']
        }
        
        # Define gender-related keywords
        gender_keywords = {
            'female_only': ['妇', '妇女', '女性', '孕妇', '妊娠', '哺乳期', '产妇', '经期', '月经', '更年期', 
                           '子宫', '卵巢', '宫颈', '阴道', '乳腺', '盆腔', '妇科', '产科', '绝经',
                           '宫内膜', '附件', '外阴', '白带', '痛经', '闭经'],
            'male_only': ['前列腺', '男性', '阳痿', '早泄', '遗精', '男科', '睾丸', '附睾', 
                         '精囊', '阴囊', '包皮', '龟头', '尿道', '精索'],
            'pregnant': ['孕妇', '妊娠', '孕期', '胎儿', '安胎', '流产', '早产', '产前', '产后'],
            'lactating': ['哺乳期', '授乳', '催乳', '回奶']
        }
        
        # Age-related checks
        if any(keyword in diagnosis for keyword in age_keywords['children']):
            if person['age'] >= 12:
                return False
        
        elif any(keyword in diagnosis for keyword in age_keywords['adult']):
            if person['age'] < 18:
                return False
        
        elif any(keyword in diagnosis for keyword in age_keywords['elderly']):
            if person['age'] < 50:
                return False
        
        # Gender-related checks
        if any(keyword in diagnosis for keyword in gender_keywords['female_only']):
            if person['gender'] != '女' and person['gender'] != 1:
                return False
        
        elif any(keyword in diagnosis for keyword in gender_keywords['male_only']):
            if person['gender'] != '男' and person['gender'] != 0:
                return False
        
        # Special population disease checks
        if any(keyword in diagnosis for keyword in gender_keywords['pregnant']):
            if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 45:
                return False
        
        elif any(keyword in diagnosis for keyword in gender_keywords['lactating']):
            if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 40:
                return False
        
        return True

    def get_diagnosis_symptom(self, person, diagnosis_dict, used_diagnosis_dict=None):
        """
        Get diagnosis and symptoms while checking diagnosis count limitations
        
        Args:
            person: Patient information
            diagnosis_dict: Diagnosis dictionary
            used_diagnosis_dict: Used diagnosis statistics dictionary
            
        Returns:
            tuple: (diagnosis name, symptoms)
        """
        keys_list = list(diagnosis_dict.keys())
        max_attempts = 50
        attempts = 0
        
        # If used_diagnosis_dict is provided, exclude diagnoses that have reached the limit when selecting
        if used_diagnosis_dict and arg.consider_coverage == 1:
            available_diagnoses = [
                diagnosis for diagnosis in keys_list 
                if used_diagnosis_dict.get(diagnosis, 0) < arg.upper_limit
            ]
            
            # If no available diagnoses, return original list (let upper layer handle)
            if not available_diagnoses:
                print(f"Warning: All diagnoses have reached the limit({arg.upper_limit}), using original diagnosis list")
                available_diagnoses = keys_list
            else:
                keys_list = available_diagnoses
                print(f"Available diagnoses: {len(available_diagnoses)}/{len(diagnosis_dict)}")
        
        while attempts < max_attempts:
            random_diagnosis = random.choice(keys_list)
            
            # Check diagnosis limitation again (double insurance)
            if used_diagnosis_dict and arg.consider_coverage == 1:
                if used_diagnosis_dict.get(random_diagnosis, 0) >= arg.upper_limit:
                    attempts += 1
                    continue
            
            if not self.check_diagnosis_reasonable(random_diagnosis, person):
                attempts += 1
                continue
            
            person['diagnosis'] = [random_diagnosis]
            symptom_result, symptom = self.llm_api.get_patient_symptom(person)
            self.llm_cache[symptom_result['input']] = symptom_result['output']
            
            if symptom:
                person['symptom'] = symptom.split('、')
                person['antecedents'] = []
                check_result, error_code = self.llm_api.check_data_error(person)
                self.llm_cache[check_result['input']] = check_result['output']
                #print(error_code)
                if error_code == '0':
                    break
            else:
                person['symptom'] = None
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"Warning: After {max_attempts} attempts, no fully qualified diagnosis found, using last result")
        
        return random_diagnosis, symptom

    def get_medicine_and_symptom(self, medicine_symptoms_dict, person, used_diagnosis_dict=None):
        """
        Get disease, symptoms and corresponding medications for patient
        
        Args:
            medicine_symptoms_dict: Disease-drug dictionary
            person: Patient information dictionary
            used_diagnosis_dict: Used diagnosis statistics dictionary
            
        Returns:
            tuple: (diagnosis name, drug list)
        """
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            try:
                diagnosis, symptom = self.get_diagnosis_symptom(person, medicine_symptoms_dict, used_diagnosis_dict)
                #print(diagnosis, symptom)
                
                medicine_data = medicine_symptoms_dict[diagnosis]
                
                drug_list = []
                for drug in medicine_data:
                    if 'drugid' in drug:
                        drug_list.append(drug['drugid'])
                
                person['medicine'] = drug_list
                
                self.check_medicine_with_KG(person)
                
                if len(person['medicine']) > 0:
                    return diagnosis, person['medicine']
                    
            except Exception as e:
                print(f"Error occurred while getting disease, symptoms and corresponding medications: {e}")
                # Don't use continue, let attempts increment and retry
                
            attempts += 1
        
        print(f"Warning: After {max_attempts} attempts, unable to get valid medications and symptoms")
        person['medicine'] = []
        return "Unknown disease", []

    def get_age(self, age_probabilities):
        """
        Generate person's age based on given age intervals and probabilities
        
        Args:
            age_probabilities: Age probability distribution data
            
        Returns:
            int: Generated age
        """
        age_ranges = list(zip(age_probabilities['age_start'], age_probabilities['age_end']))
        probabilities = age_probabilities['probability'].values
        num_ranges = len(age_ranges)  # Dynamically get number of ranges
        index = random.choices(range(num_ranges), weights=probabilities)[0]
        age_range = age_ranges[index]
        age = random.randint(age_range[0], age_range[1])
        return age

    def check_medicine_reasonable(self, medicine_list, person):
        """
        Filter out medications that don't match patient's gender and age based on literal meaning
        
        Args:
            medicine_list: Medicine list
            person: Patient information
        
        Returns:
            list: Filtered medicine list
        """
        filtered_medicines = []
        
        # Define keyword rules
        age_keywords = {
            'children': ['小儿', '儿童', '婴儿', '新生儿', '幼儿', '婴幼儿', '小孩'],
            'adult': ['成人'],
            'elderly': ['老年', '老人']
        }
        
        gender_keywords = {
            'female_only': ['妇', '妇女', '女性', '孕妇', '妊娠', '哺乳期', '授乳', '产妇', '经期', '月经', '更年期'],
            'male_only': ['壮阳', '前列腺', '男性', '阳痿', '早泄', '遗精', '补肾壮阳', '男科'],
            'pregnant': ['孕妇', '妊娠', '孕期', '胎儿', '安胎'],
            'lactating': ['哺乳期', '授乳', '催乳', '回奶']
        }
        
        for medicine in medicine_list:
            medicine_name = medicine.get('drug', '') if isinstance(medicine, dict) else str(medicine)
            should_include = True
            
            # Age-related checks
            if any(keyword in medicine_name for keyword in age_keywords['children']):
                if person['age'] >= 12:
                    should_include = False
            
            elif any(keyword in medicine_name for keyword in age_keywords['adult']):
                if person['age'] < 18:
                    should_include = False
            
            elif any(keyword in medicine_name for keyword in age_keywords['elderly']):
                if person['age'] < 65:
                    should_include = False
            
            # Gender-related checks
            if should_include:
                if any(keyword in medicine_name for keyword in gender_keywords['female_only']):
                    if person['gender'] != '女' and person['gender'] != 1:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in gender_keywords['male_only']):
                    if person['gender'] != '男' and person['gender'] != 0:
                        should_include = False
            
            # Special population checks
            if should_include:
                if any(keyword in medicine_name for keyword in gender_keywords['pregnant']):
                    if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 45:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in gender_keywords['lactating']):
                    if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 40:
                        should_include = False
            
            # Additional medicine name checks
            if should_include:
                if any(keyword in medicine_name for keyword in ['避孕', '紧急避孕']):
                    if (person['gender'] != '女' and person['gender'] != 1) or person['age'] < 18 or person['age'] > 50:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in ['雌激素', '雌二醇', '黄体酮']):
                    if person['gender'] != '女' and person['gender'] != 1:
                        should_include = False
                
                elif any(keyword in medicine_name for keyword in ['睾酮', '雄激素']):
                    if person['gender'] != '男' and person['gender'] != 0:
                        should_include = False
            
            if should_include:
                filtered_medicines.append(medicine)
        
        return filtered_medicines

    def add_antecedents_and_on_medicine(self, person):
        """
        Generate patient's medical history antecedents and current medications on_medicine
        
        Args:
            person: Patient information dictionary, must contain 'medicine' field
        """
        try:
            #reviewer = DrugReviewSystem(self.graph.service.uri)
            
            if not self._interaction_dict or not self._drugmsg_dict:
                self._load_data_files()
                
        except Exception as e:
            print(f"Error: Problem occurred during data initialization - {e}")
            person['antecedents'] = []
            person['on_medicine'] = []
            return

        if 'medicine' not in person or not isinstance(person['medicine'], list):
            print("Warning: Patient drug information does not exist or format is incorrect")
            person['antecedents'] = []
            person['on_medicine'] = []
            return
            
        gold_answer_list = person['medicine'].copy()
        
        # 1. Filter out drug ID list from gold_answers that have interaction records in interaction_dict
        medicines_with_interactions = []
        for medicine_id in gold_answer_list:
            interact_drugs = self._interaction_dict.get(medicine_id, {}).get("interaction_drug", [])
            if interact_drugs:  # If there are interaction records
                medicines_with_interactions.append(medicine_id)
        
        n = len(medicines_with_interactions)  # Total number of drugs with interaction records
        print(f"Total number of drugs with interaction records: {n}")
        if n == 0:
            person['antecedents'] = []
            person['on_medicine'] = []
            person['medicine'] = gold_answer_list
            return
        
        # Loop logic
        j = 0  # Initially selected j items
        antecedents = []
        on_medicine = []
        selected_medicines = []  # Record selected medicines
        selected_interact_drugs = set()  # Record selected interaction drug IDs to avoid duplicates
        
        max_attempts = 20  # Maximum loop attempts to avoid infinite loop
        attempt_count = 0
        
        while j < n and attempt_count < max_attempts:
            attempt_count += 1
            
            # 2. If n-j>0, randomly generate a number m from [1,n-j], select m medicines
            if n - j > 0:
                m = random.randint(1, n - j)
                
                # Select m from unselected medicines
                available_medicines = [med for med in medicines_with_interactions if med not in selected_medicines]
                if len(available_medicines) < m:
                    m = len(available_medicines)
                
                if m == 0:
                    break
                    
                current_selected = random.sample(available_medicines, m)
                selected_medicines.extend(current_selected)
                j += m  # Update selected count
                
                # 3. Get interaction drugs for m medicines, select one on_medicine for each medicine
                current_on_medicine = []
                current_antecedents = []
                
                for selected_medicine_id in current_selected:
                    try:
                        interact_drugs = self._interaction_dict.get(int(selected_medicine_id), {}).get("interaction_drug", [])
                        if not interact_drugs:
                            continue
                       
                        # First shuffle interaction drug list, then select in order to avoid duplicates
                        shuffled_interact_drugs = interact_drugs.copy()
                        random.shuffle(shuffled_interact_drugs)
                        
                        selected_interact_drug = None
                        interact_drug_id = None
                        
                        # Select first unselected interaction drug in order
                        for drug in shuffled_interact_drugs:
                            potential_drug_id = drug.get("id")
                        
                            if potential_drug_id and potential_drug_id not in selected_interact_drugs:
                                selected_interact_drug = drug
                                interact_drug_id = potential_drug_id
                                selected_interact_drugs.add(interact_drug_id)  # Record selected drug
                                break
                        
                        if not interact_drug_id:
                            continue
                        
                        current_on_medicine.append(interact_drug_id)
                        
                        # 4. Select a main symptom of on_medicine as medical history
                        drugmsg = self._drugmsg_dict.get(int(interact_drug_id))
                        if not drugmsg:
                            continue
                        
                        treat_list = drugmsg.get('治疗', [])
                        if not treat_list:
                            continue
                        
                        k = len(treat_list)
                        max_symptom_attempts = min(k, 20)
                        selected_symptom = None
                        
                        # Try to find a reasonable symptom
                        tried_symptoms = set()  # Use set to improve search efficiency
                        for attempt in range(max_symptom_attempts):
                            if len(tried_symptoms) >= k:  # All symptoms have been tried
                                break

                            # If attempts exceed symptom count, randomly select a symptom
                            if max_symptom_attempts < len(treat_list):
                                candidate_symptom = random.choice(treat_list)
                            else:
                                # If attempts are less than symptom count, select symptoms in order
                                candidate_symptom = treat_list[attempt]

                            if candidate_symptom in tried_symptoms:
                                continue
                            
                            tried_symptoms.add(candidate_symptom)
                            
                            if self.check_diagnosis_reasonable(candidate_symptom, person):
                                selected_symptom = candidate_symptom
                                break
                        
                        # If a reasonable symptom is found, add to medical history
                        if selected_symptom:
                            current_antecedents.append(selected_symptom)
                            
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"Error occurred while processing drug {selected_medicine_id}: {e}")
                        continue
                
                # 5. If 0 medical history, return to step 2 and continue loop
                if len(current_antecedents) == 0:
                    continue  # Continue loop
                
                # 6. If there are [1, m] medical histories, add current results to total results
                antecedents.extend(current_antecedents)
                on_medicine.extend(current_on_medicine)
                
                # Successfully found medical history, can end loop
                break
            else:
                break
        
        # Set initial results
        person['antecedents'] = antecedents
        person['on_medicine'] = on_medicine
        
        # 6. LLM checks if medical history matches patient information
        if antecedents:  # Only check when there is medical history
            try:
                result, error_code = self.llm_api.check_data_error(person)
                #print(error_code)
                if error_code != '0':
                    person['antecedents'] = []
                    person['on_medicine'] = []
                    print("LLM determined medical history doesn't match patient characteristics, cleared medical history")
            except Exception as e:
                print(f"Error occurred during LLM validation: {e}")
                # Conservative handling when error occurs, clear medical history
                person['antecedents'] = []
                person['on_medicine'] = []
        
        # 7. Use interaction_check to check and remove drugs that interact with on_medicine
        if person['on_medicine'] and gold_answer_list:
            try:
                drugs_to_remove = []
                for gold_drug_id in gold_answer_list:
                    for on_drug_id in person['on_medicine']:
                        if self.reviewer.interaction_check(int(gold_drug_id),int(on_drug_id)):
                            drugs_to_remove.append(gold_drug_id)
                            break
                
                # Remove drugs with interactions
                for drug_id in drugs_to_remove:
                    if drug_id in gold_answer_list:
                        gold_answer_list.remove(drug_id)
                        
            except Exception as e:
                print(f"Error occurred while checking drug interactions: {e}")
        
        person['medicine'] = gold_answer_list

    def read_all_msg(self):
        """Read all cached messages"""
        diagnosis_dict = {}
        
        # Fix file paths, read correct data files
        people_file_path = f"output/{arg.history_doc}/people_data.pkl"
        llm_cache_file_path = f"output/{arg.history_doc}/LLM_cache.pkl"
        diagnosis_file_path = f"output/{arg.history_doc}/used_diagnosis_dict.json"
        
        people_list = []
        llm_cache = {}
        
        # Read patient data
        try:
            with open(people_file_path, "rb") as fp:
                people_list = pickle.load(fp)
            print(f"✓ Successfully read {len(people_list)} existing patient data")
        except FileNotFoundError:
            print(f"⚠ Existing patient data file not found: {people_file_path}")
            people_list = []
        except Exception as e:
            print(f"❌ Error reading patient data: {e}")
            people_list = []
        
        # Read LLM cache
        try:
            with open(llm_cache_file_path, "rb") as f:
                llm_cache = pickle.load(f)
            print(f"✓ Successfully read LLM cache with {len(llm_cache)} records")
        except FileNotFoundError:
            print(f"⚠ LLM cache file not found: {llm_cache_file_path}")
            llm_cache = {}
        except Exception as e:
            print(f"❌ Error reading LLM cache: {e}")
            llm_cache = {}
        
        # Read diagnosis statistics
        try:
            with open(diagnosis_file_path, 'r', encoding='utf-8') as f:
                diagnosis_dict = json.load(f)
            print(f"✓ Successfully read diagnosis statistics with {len(diagnosis_dict)} diagnoses")
        except FileNotFoundError:
            print(f"⚠ Diagnosis statistics file not found: {diagnosis_file_path}")
            # Rebuild diagnosis statistics from existing patient data
            for person in people_list:
                if 'diagnosis' in person:
                    diagnosis_list = person['diagnosis'] if isinstance(person['diagnosis'], list) else [person['diagnosis']]
                    for diagnosis in diagnosis_list:
                        if diagnosis in diagnosis_dict:
                            diagnosis_dict[diagnosis] += 1
                        else:
                            diagnosis_dict[diagnosis] = 1
        except Exception as e:
            print(f"❌ Error reading diagnosis statistics: {e}")
            diagnosis_dict = {}
        
        return people_list, llm_cache, diagnosis_dict
    
    def check_medicine_with_KG(self, person):
        """
        Review medications through knowledge graph, remove unqualified drugs
        
        Args:
            person: Patient information including medicine, age, group, allergen fields
        """
        if not person.get('medicine'):
            return
            
        medicines_to_remove = set()
        
        # Age review
        age_pass, age_failed_medicines = self.reviewer.age_review(person['medicine'], person['age'])
        if not age_pass:
            medicines_to_remove.update(age_failed_medicines)
        
        # Special population review
        if 'group' in person:
            population_pass, population_failed_medicines = self.reviewer.special_population_review(
                person['medicine'], person['group']
            )
            if not population_pass:
                medicines_to_remove.update(population_failed_medicines)
        
        # Allergen review
        allergen = person.get('allergen', [])
        if allergen and allergen != '无' and allergen != ['无']:
            allergen_pass, allergen_failed_medicines = self.reviewer.allergy_review(
                person['medicine'], allergen
            )
            if not allergen_pass:
                medicines_to_remove.update(allergen_failed_medicines)
        # Check if removal is correct
        #print(person['medicine'])
        #print(medicines_to_remove)
        # Remove unqualified drugs from medicine list
        person['medicine'] = [med for med in person['medicine'] 
                             if med not in medicines_to_remove]
        #print(person['medicine'])

    def decide_group(self, person):
        """
        Determine population group based on age
        Children (<12); Adolescents (>=12, <18); Adults (>=18, <65); Elderly (>=65)
        Add liver dysfunction and kidney dysfunction based on probability
        For females, add pregnant or lactating labels based on probability
        
        Args:
            person: Patient information dictionary
        """
        age = person['age']
        gender = person.get('gender', '')
        group_list = []
        
        # Basic age grouping
        if age < 12:
            group_list.append('儿童')
        elif 12 <= age < 18:
            group_list.append('青少年')
        elif 18 <= age < 65:
            group_list.append('成人')
        else:
            group_list.append('老年人')
        
        # Add special populations based on probability
        if random.random() < arg.liver_prob:
            group_list.append('肝功能不全')
        
        if random.random() < arg.kidney_prob:
            group_list.append('肾功能不全')
        
        # Special female population handling
        if gender == '女' or gender == 1:
            # Only women of childbearing age can be pregnant or lactating (18-45 years old)
            if 18 <= age <= 45:
                # Randomly choose whether to be pregnant or lactating (mutually exclusive)
                special_female_choice = random.random()
                
                if special_female_choice < arg.pregnant_prob:
                    group_list.append('孕妇')
                elif special_female_choice < arg.pregnant_prob + arg.lactation_prob:
                    group_list.append('哺乳期')
                # If neither is met, maintain normal female status
        
        person['group'] = group_list

    def decide_gender(self, person):
        """
        Determine gender information based on gender number
        0 for male, 1 for female
        
        Args:
            person: Patient information dictionary
        """
        gender_id = person['gender']
        
        if gender_id == 0:
            person['gender'] = '男'
        elif gender_id == 1:
            person['gender'] = '女'

    def get_medicine_msg(self, medicine_list):
        """
        Get medicine information list
        
        Args:
            medicine_list: Drug ID list
            
        Returns:
            list: Drug information list
        """
        drug_msg_list = []
        for medicine in medicine_list:
            drug_msg = self.get_drugmsg_from_mkg(medicine)
            drug_msg_list.append(drug_msg)
        return drug_msg_list

    def get_drugmsg_from_mkg(self, drugid):
        """
        Get drug information from knowledge graph
        
        Args:
            drugid: Drug ID
            
        Returns:
            dict: Drug information record
        """
        search = self.graph.run(
            """
            MATCH (drug:`药品`)
            WHERE id(drug) = $drugid
            
            WITH drug, drug.name AS name, drug.number AS CMAN
            OPTIONAL MATCH p1=(drug)-[:用药*0..2]->(fact:`知识组`)-[:用药]->(crowd:`人群`),
                            p2=(fact)-[:用药结果]->(useResult:`用药结果级别`)
            WITH drug, name, CMAN, 
                    collect(DISTINCT {crowdid: id(crowd), crowd: crowd.name, useresultid: id(useResult), useresult: useResult.name}) AS crowdInfo
            
            OPTIONAL MATCH p3=(drug)-[:治疗*0..3]->(treatment:`病症`)
            WITH drug, name, CMAN, crowdInfo,
                    collect(DISTINCT {treatid: id(treatment), treat: treatment.name}) AS treatmentInfo
            
            OPTIONAL MATCH p4=(drug)-[:成分*0..3]->(ingre:`药物`)
            WITH drug, name, CMAN, crowdInfo, treatmentInfo,
                    collect(DISTINCT {ingredientId: id(ingre), ingredient: ingre.name}) AS ingredients
            
            OPTIONAL MATCH p5=(drug)-[:相互作用*0..3]->(inter:`药物`)
            WITH drug, name, CMAN, crowdInfo, treatmentInfo, ingredients,
                    collect(DISTINCT {interactionId: id(inter), interaction: inter.name}) AS interactions
            
            RETURN name, CMAN, crowdInfo, treatmentInfo, ingredients, interactions
            """, drugid=drugid)
        result = search.data()[0]
        
        # Process knowledge graph returned results
        caution = []
        for item in result["crowdInfo"]:
            temp = {
                "crowd_id": str(item["crowdid"]),
                "crowd": item["crowd"],
                "caution_levelid": item["useresultid"],
                "caution_level": item["useresult"]
            }
            caution.append(temp)
        
        treat = []
        for item in result["treatmentInfo"]:
            temp = {
                "treat_id": str(item["treatid"]),
                "treat": item["treat"]
            }
            treat.append(temp)
        
        ingredients_list = []
        for item in result["ingredients"]:
            temp = {
                "ingredient_id": str(item['ingredientId']),
                "ingredient": item["ingredient"],
            }
            ingredients_list.append(temp)
        
        interaction_list = []
        for item in result["interactions"]:
            temp = {
                "interaction_id": str(item['interactionId']),
                "interaction": item["interaction"],
            }
            interaction_list.append(temp)
            
        if result["interactions"][0]["interaction"] is None:
            interaction_list = []
        if result["ingredients"][0]["ingredient"] is None:
            ingredients_list = []
        if result["crowdInfo"][0]["crowd"] is None:
            caution = []
            
        record = {
            "drugid": str(drugid),
            "name": result["name"],
            "CMAN": result["CMAN"],
            "treat": treat,
            "caution": caution,
            "ingredients": ingredients_list,
            "interaction": interaction_list
        }
        
        return record

    def generate_people_data(self, num):
        """
        Generate population data
        
        Args:
            num: Number of patients to generate
            
        Returns:
            list: List of generated patient data
        """
        # Initialize data structures
        diagnosis_dict = {}
        people_list = []
        start_id = 0
        
        # If need to generate based on historical data
        if arg.history_data == 1:
            print(f"Continue generation mode: Generate additional {num} patient data based on existing data...")
            people_list, self.llm_cache, diagnosis_dict = self.read_all_msg()
            
            # Find maximum ID in existing data, then increment from there
            if people_list:
                # Convert string IDs to integers to find maximum, handle possible invalid IDs
                valid_ids = []
                for person in people_list:
                    try:
                        person_id = person.get('id', '0')
                        if isinstance(person_id, (int, float)):
                            valid_ids.append(int(person_id))
                        elif isinstance(person_id, str) and person_id.isdigit():
                            valid_ids.append(int(person_id))
                        else:
                            print(f"Warning: Found invalid ID format: {person_id}")
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Error converting ID: {person.get('id')} - {e}")
                
                if valid_ids:
                    max_existing_id = max(valid_ids)
                    start_id = max_existing_id + 1
                    print(f"✓ Read {len(people_list)} existing data, maximum valid ID is {max_existing_id}, new data will start from ID {start_id}")
                else:
                    start_id = 0
                    print(f"⚠ No valid IDs found in existing data, new data will start from ID {start_id}")
            else:
                start_id = 0
                print(f"✓ No existing data found, new data will start from ID {start_id}")
        else:
            print(f"Regeneration mode: Generate {num} brand new patient data...")

        # Create output directory
        if not os.path.exists(f"output/{arg.out_doc}"):
            os.makedirs(f"output/{arg.out_doc}")
        
        # Load configuration data
        self._load_data_files()

        # Diagnosis coverage monitoring variables
        total_diagnosis_count = len(self._medicine_symptoms_dict)
        consecutive_skips = 0  # Consecutive skip count
        max_consecutive_skips = 100  # Maximum consecutive skips

        # Generate patient data
        mode_text = "Continue generation" if arg.history_data == 1 else "Generation"
        print(f"\nStart {mode_text} {num} patient information...")
        print(f"Diagnosis limitation settings: consider_coverage={arg.consider_coverage}, upper_limit={arg.upper_limit}")
        print(f"Total available diagnoses: {total_diagnosis_count}")
        now_num = 0
        pbar = tqdm(total=num)
        
        while now_num < num:
            person = {}
            person['id'] = str(start_id + now_num)  # Ensure ID is string type
            try:
                # 1. Generate basic information: age and gender
                person['age'] = self.get_age(self._age_probabilities)
                person['gender'] = random.randint(0, 1)
                # 2. Convert gender number to text (0->male, 1->female)
                self.decide_gender(person)
                
                # 3. Determine population group (based on age and special situations)
                self.decide_group(person)
                
                # 4. Initialize allergens (needed before drug checks)
                if random.random() < arg.allergen_prob:
                    person['allergen'] = [random.choice(self._allergen_list)]
                else:
                    person['allergen'] = []
                
                # 5. Get disease, symptoms, drugs (diagnosis limitation check completed inside function)
                diagnosis, medicine = self.get_medicine_and_symptom(self._medicine_symptoms_dict, person, diagnosis_dict)
                
                # 6. Initialize medical history and concurrent medications as empty lists
                person['antecedents'] = []
                person['on_medicine'] = []
                
                # 7. Based on probability determine whether to add special medical history and concurrent medications
                if random.random() < arg.medhistory_prob:
                    self.add_antecedents_and_on_medicine(person)
                
                # 8. Convert drug IDs to detailed drug information
                person['medicine'] = self.get_medicine_msg(person['medicine'])
                person['on_medicine'] = self.get_medicine_msg(person['on_medicine'])
                
                # 9. Update diagnosis statistics
                if diagnosis in diagnosis_dict:
                    diagnosis_dict[diagnosis] += 1
                else:
                    diagnosis_dict[diagnosis] = 1
                
                # 10. Add to result list
                people_list.append(person)
                now_num += 1
                pbar.update(1)
                
                # Reset consecutive skip counter
                consecutive_skips = 0
                
            except Exception as e:
                print(f"Error occurred while generating patient {start_id + now_num + 1}: {e}")
                # Increase skip count when exception occurs
                consecutive_skips += 1
                
                # Check if too many consecutive errors
                if consecutive_skips >= max_consecutive_skips:
                    print(f"\nError: {max_consecutive_skips} consecutive generation failures")
                    print(f"Currently generated {now_num} patients, target {num}")
                    
                    # Ask user whether to continue
                    response = input("Do you want to continue generation? (y/n): ")
                    if response.lower() != 'y':
                        print("Stopping generation, saving current results...")
                        break
                    else:
                        consecutive_skips = 0  # Reset counter
                        print("Continuing generation...")
                continue
        
        pbar.close()
        
        # Display diagnosis coverage statistics
        print(f"\n=== Diagnosis Coverage Statistics ===")
        print(f"Total available diagnoses: {total_diagnosis_count}")
        print(f"Actually used diagnoses: {len(diagnosis_dict)}")
        coverage_rate = len(diagnosis_dict)/total_diagnosis_count*100 if total_diagnosis_count > 0 else 0
        print(f"Diagnosis coverage rate: {coverage_rate:.2f}%")
        
        # Group statistics by usage count
        usage_stats = {}
        for diagnosis, count in diagnosis_dict.items():
            if count not in usage_stats:
                usage_stats[count] = 0
            usage_stats[count] += 1
        
        print(f"Diagnosis usage count distribution:")
        for usage_count in sorted(usage_stats.keys()):
            print(f"  Diagnoses used {usage_count} times: {usage_stats[usage_count]} items")
        
        # Generate analysis report
        print("\nGenerating statistical analysis report...")
        self.data_analyzer.age_analysis(people_list)
        self.data_analyzer.gender_analysis(people_list)
        self.data_analyzer.group_analysis(people_list)
        
        # Save data
        print("\nSaving data files...")
        
        # Save diagnosis usage statistics
        with open(f"output/{arg.out_doc}/used_diagnosis_dict.json", 'w', encoding='utf-8') as f:
            json.dump(diagnosis_dict, f, ensure_ascii=False, indent=4)
        
        # Save LLM cache
        with open(f"output/{arg.out_doc}/LLM_cache.pkl", "wb") as fp:
            pickle.dump(self.llm_cache, fp)
        
        # Save complete patient data list - PKL format
        with open(f"output/{arg.out_doc}/people_data.pkl", "wb") as fp:
            pickle.dump(people_list, fp)
        print(f"✓ Patient data saved to: output/{arg.out_doc}/people_data.pkl")
        
        # Save detailed diagnosis statistics report
        diagnosis_report = {
            "Summary": {
                "Total available diagnoses": total_diagnosis_count,
                "Actually used diagnoses": len(diagnosis_dict),
                "Diagnosis coverage rate": f"{coverage_rate:.2f}%",
                "Total generated patients": len(people_list),
                "upper_limit setting": arg.upper_limit
            },
            "Usage count distribution": usage_stats,
            "Detailed usage statistics": diagnosis_dict
        }
        
        with open(f"output/{arg.out_doc}/diagnosis_coverage_report.json", 'w', encoding='utf-8') as f:
            json.dump(diagnosis_report, f, ensure_ascii=False, indent=4)
        
        print(f"✓ Diagnosis coverage report saved to: output/{arg.out_doc}/diagnosis_coverage_report.json")
        
        # Save complete patient data list - JSON format  
        try:
            with open(f"output/{arg.out_doc}/people_data.json", 'w', encoding='utf-8') as f:
                json.dump(people_list, f, ensure_ascii=False, indent=2)
            print(f"✓ Patient data saved to: output/{arg.out_doc}/people_data.json")
        except (TypeError, ValueError) as e:
            print(f"⚠ JSON save failed, may contain non-serializable objects: {e}")
        
        if arg.history_data == 1:
            print(f"\n✅ Successfully generated additional {num} patient data based on existing {start_id} data!")
            print(f"Now have a total of {len(people_list)} patient data")
        else:
            print(f"\n✅ Successfully generated and saved {len(people_list)} patient data!")
        
        print(f"Data file locations:")
        print(f"  - PKL format: output/{arg.out_doc}/people_data.pkl")
        print(f"  - JSON format: output/{arg.out_doc}/people_data.json")
        print(f"  - Diagnosis statistics: output/{arg.out_doc}/used_diagnosis_dict.json")
        print(f"  - Diagnosis coverage report: output/{arg.out_doc}/diagnosis_coverage_report.json")
        print(f"  - LLM cache: output/{arg.out_doc}/LLM_cache.pkl")
        
        return people_list

    def load_people_data(self, file_format="pkl"):
        """
        Load patient data from file
        
        Args:
            file_format: File format, "pkl" or "json"
            
        Returns:
            list: Patient data list
        """
        if file_format.lower() == "pkl":
            file_path = f"output/{arg.out_doc}/people_data.pkl"
            try:
                with open(file_path, "rb") as fp:
                    people_data = pickle.load(fp)
                print(f"✓ Successfully loaded {len(people_data)} patient data from PKL file")
                return people_data
            except FileNotFoundError:
                print(f"❌ File not found: {file_path}")
                return []
            except Exception as e:
                print(f"❌ Error loading PKL file: {e}")
                return []
                
        elif file_format.lower() == "json":
            file_path = f"output/{arg.out_doc}/people_data.json"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    people_data = json.load(f)
                print(f"✓ Successfully loaded {len(people_data)} patient data from JSON file")
                return people_data
            except FileNotFoundError:
                print(f"❌ File not found: {file_path}")
                return []
            except Exception as e:
                print(f"❌ Error loading JSON file: {e}")
                return []
        else:
            print(f"❌ Unsupported file format: {file_format}, please use 'pkl' or 'json'")
            return []

    @staticmethod
    def get_data_summary(people_data):
        """
        Get basic statistical summary of patient data
        
        Args:
            people_data: Patient data list
            
        Returns:
            dict: Statistical summary
        """
        if not people_data:
            return {"error": "Data is empty"}
        
        total_count = len(people_data)
        ages = [person['age'] for person in people_data]
        genders = [person['gender'] for person in people_data]
        
        summary = {
            "Total population": total_count,
            "Age statistics": {
                "Minimum age": min(ages),
                "Maximum age": max(ages),
                "Average age": round(sum(ages) / len(ages), 1)
            },
            "Gender distribution": {
                "Male": genders.count('男'),
                "Female": genders.count('女')
            },
            "Sample fields": list(people_data[0].keys()) if people_data else []
        }
        
        return summary
    

def load_people_data(file_format="pkl"):
    """
    Load patient data from file
    
    Args:
        file_format: File format, "pkl" or "json"
    """
    synthetic = Synthetic()
    people_data = synthetic.load_people_data(file_format)
    print(Synthetic.get_data_summary(people_data))
    return people_data


def generate_people_data(num):
    """
    Wrapper function compatible with original interface
    
    Args:
        num: Number of patients to generate
        
    Returns:
        list: List of generated patient data
    """
    synthetic = Synthetic()
    return synthetic.generate_people_data(num)



if __name__ == '__main__':
    generate_people_data(arg.people_num)
    
    