import pickle
import os
import json
import csv
import random
import dashscope
import re

class LLMAPI:
    """
    Large Language Model API class, provides patient symptom acquisition and data error checking functionality
    """
    
    def __init__(self, api_key, model="qwen-max"):
        """
        Initialize LLMAPI class
        
        Args:
            api_key (str): API key
            model (str): Model name, default is qwen-max
        """
        self.api_key = api_key
        self.model = model
        
    def _get_patient_prompt(self, item, spliter=" || "):
        """
        Get patient symptom prompt template
        
        Args:
            item (dict): Patient data item
            spliter (str): Separator
            
        Returns:
            tuple: (prompt string, input information)
        """
        def format_input(item, spliter=" || "):
            age = item["age"]
            gender = item["gender"]
            group = ",".join(item["group"])
            diagnosis = ",".join(item["diagnosis"])
            return f"{age}{spliter}{gender}{spliter}{group}{spliter}XX{spliter}{diagnosis}"

        input_msg = format_input(item, spliter)

        SYMPTOM_PROMPT = """请根据病人信息和诊断给出合理的主诉症状,按照该格式输出:年龄 || 性别 || 人群类别 || XX || 诊断。其中年龄、性别、人群类别、诊断由我给出，你只需在XX处填入主诉症状信息。
            请注意,你只需在XX处生成主诉症状信息,不要输出任何别的信息或者提示,不要修改我已经给出的其他信息.下面给出8个例子:

            1.input:35岁 || 男 || 成人 || XX || 呼吸道感染
            output:35岁 || 男 || 成人 || 咳嗽、咳痰、发热 || 呼吸道感染

            2.input:29岁 || 女 || 成人 || XX || 尿路感染
            output:29岁 || 女 || 成人 || 尿频、尿急、尿痛 || 尿路感染

            3.intput:33岁 || 女 || 哺乳期 || XX || 乳腺炎
            output:33岁 || 女 || 哺乳期 || 乳房疼痛、红肿

            4.intput:26岁 || 女 || 孕妇 || XX || 发烧
            output:26岁 || 女 || 孕妇 || 发热、头痛、乏力 || 发烧

            5.intput:42岁 || 男 || 成人 || XX || 前列腺炎
            output:42岁 || 男 || 成人 || 尿频、尿急、尿痛 || 前列腺炎

            6.intput:7岁 || 女 || 儿童 || XX || 消化不良
            output:7岁 || 女 || 儿童 || 食欲不振、腹胀、腹泻 || 消化不良

            7.input:67岁 || 男 || 老年人 || XX || 高血压
            output:67岁 || 男 || 老年人 || 头晕、心悸、胸闷 || 高血压

            8.input:70岁 || 男 || 老年人 || XX || 骨质疏松症
            output:70岁 || 男 || 老年人 || 腰背疼痛、易骨折 || 骨质疏松症

            下面给出病人信息和诊断，请按照格式输出。\n
            input:
        """

        prompt = SYMPTOM_PROMPT + input_msg

        return prompt, input_msg

    def _get_error_check_prompt(self, item, spliter=" || "):
        """
        Get error check prompt template
        
        Args:
            item (dict): Data item
            spliter (str): Separator
            
        Returns:
            tuple: (prompt string, input information)
        """
        def format_input_data(item, spliter=" || "):
            """
            Format input data
            
            Args:
                item (dict): Data item
                spliter (str): Separator
            
            Returns:
                str: Formatted input string
            """
            age = item["age"]
            group = ",".join(item["group"])
            gender = item["gender"]
            symptom = ",".join(item["symptom"])
            diagnosis = ",".join(item["diagnosis"])
            antecedents = ",".join(item["antecedents"])
            
            if not item["antecedents"]:
                return f"{age}{spliter}{group}{spliter}{gender}{spliter}{symptom}{spliter}{diagnosis}{spliter}无既往病史"
            else:
                return f"{age}{spliter}{group}{spliter}{gender}{spliter}{symptom}{spliter}{diagnosis}{spliter}{antecedents}"
        
        input_msg = format_input_data(item)

        ERRORCHECK_PROMPT = """你是一个专业的医生，你的任务是判断病历中是否存在错误。如果存在错误，请根据错误类型编号返回数字；如果没有错误，请返回 0。

        病历的格式为：
        年龄 || 人群 || 性别 || 症状 || 疾病 || 既往病史

        错误类型包括：
        1. 疾病与性别不符；
        2. 疾病与年龄不符；
        3. 疾病与症状不符；
        4. 疾病描述不规范；
        5. 既往病史与性别不符；
        6. 既往病史与年龄不符；
        7. 既往病史描述不规范；
        0. 表示没有错误。

        请严格按照以下格式输出：
        输出：[错误编号](多个错误时用逗号分隔，如:1,5）

        给出以下例子：
        输入: 35 || 孕妇,肾功能不全 || 女 || 动脉粥状硬化 || 胸痛,呼吸困难,水肿 || 无既往病史
        输出: 2

        输入: 80 || 老年人 || 女 || 月经不调,经血颜色改变 || 经色紫暗 || 食少便溏
        输出: 2

        输入: 37 || 孕妇 || 女 || 乳房胀痛,乳头溢液呈豆渣状 || 豆渣状 || 顽痘
        输出: 4

        输入: 47 || 成人 || 男 || 疲劳感,精力不足,性欲减退 || 需求增加 || 黄褐斑
        输出: 4

        输入: 20 || 成人，哺乳期 || 女 || 恶心,呕吐,头痛,乏力 || 抗辐射 || 缺乏症
        输出: 3

        输入: 1 || 儿童,肾功能不全 || 女 || 发育迟缓,肌张力异常 || 机能障碍 || 黄体功能不足
        输出: 6

        输入: 27 || 成人 || 女 || 呼吸急促,胸闷,喉部喘鸣声 || 咳嗽,喘鸣 || 小面积
        输出: 7

        输入: 41 || 成人 || 男 || 避孕咨询,焦虑 || 女性避孕 || 重度持续性哮喘
        输出: 1

        输入: 29 || 成人|| 男 || 头皮瘙痒,鳞屑增多 || 头皮鳞屑 || 盆腔炎
        输出: 5

        输入: 53 || 成人 || 男 || 右上腹痛,乏力,食欲不振 || 肝功能不正常 || 疾病
        输出: 7

        输入: 52 || 成人 || 男 || 胸痛,烧心感 || 吞酸 || 更年期综合征
        输出: 5

        输入: 48 || 成人 || 男 || 乳房发育不良,无乳汁分泌 || 无乳 || 跌打损伤
        输出: 1

        输入: 41 || 成人 || 男 || 胸痛,心悸,气促 || 不稳定型冠状动脉疾病 || 晚期卵巢癌
        输出: 5

        输入: 38 || 成人,孕妇 || 女 || 皮肤瘙痒,光敏感性皮疹,乏力 || 皮肤卟啉病 || 黄褐斑
        输出: 0

        输入: 59 || 成人 || 男 || 阴茎勃起功能障碍,潮热,情绪波动 || 绝经 || 闭经
        输出: 1,5

        输入: """
        
        prompt = ERRORCHECK_PROMPT + input_msg
        return prompt, input_msg

    def _call_llm_api(self, prompt):
        """
        Call LLM API
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Model returned result
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = dashscope.Generation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                temperature=0,
                top_p=0.01,
                result_format='message',
            )
            
            output = response.output.choices[0].message.content
            return output
            
        except Exception as e:
            print(f"Error message: {e}")
            print("Please refer to documentation: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            return f"ERROR: {str(e)}"

    def get_patient_symptom(self, item):
        """
        Get patient symptoms
        
        Args:
            item (dict): Patient data item, contains age, gender, group, diagnosis fields
            
        Returns:
            str: Extracted symptom information
        """
        prompt, input_msg = self._get_patient_prompt(item)
        output = self._call_llm_api(prompt)
        
        print(f"id: {item.get('id', 'N/A')}, output: {output}")
        
        # Extract symptom part
        symptom = self.extract_symptom_from_output(output)
        result = {
            'id': item.get('id', 'N/A'),
            'input': input_msg,
            'output': output,
            'symptom': symptom
        }
        return result, symptom

    def check_data_error(self, item, max_retries=3):
        """
        Check data errors
        
        Args:
            item (dict): Data item, contains age, group, gender, symptom, diagnosis, antecedents fields
            max_retries (int): Maximum retry attempts
            
        Returns:
            tuple: (dictionary containing check results, error code string)
        """
        prompt, input_msg = self._get_error_check_prompt(item)
        
        for attempt in range(max_retries + 1):
            output = self._call_llm_api(prompt)
            print(f"id: {item.get('id', 'N/A')}, attempt: {attempt + 1}, output: {output}")
            
            # Try to parse error code
            error_code = self._extract_error_code(output)
            
            if error_code is not None:
                # Successfully parsed, return result
                result = {
                    'id': item.get('id', 'N/A'),
                    'input': input_msg,
                    'output': output,
                    'attempts': attempt + 1
                }
                return result, error_code
            else:
                # Parse failed, record warning
                print(f"Warning: LLM output format incorrect (attempt {attempt + 1}/{max_retries + 1}): {output}")
                
                if attempt < max_retries:
                    # Not the last attempt, call LLM again
                    prompt = self._add_format_reminder_to_prompt(prompt)
                    continue
        
        # All attempts failed, return default value
        print(f"Error: After {max_retries + 1} attempts, still unable to parse LLM output format, defaulting to error code '0'")
        result = {
            'id': item.get('id', 'N/A'),
            'input': input_msg,
            'output': output,
            'attempts': max_retries + 1,
            'parse_failed': True
        }
        return result, '0'
    
    def _extract_error_code(self, output):
        """
        Safely extract error code from LLM output
        
        Args:
            output (str): LLM raw output
            
        Returns:
            str or None: Error code string, returns None if parsing fails
        """
        import re
        
        # Method 1: Find content after "Output:" or "输出："
        patterns = [
            r'输出[：:]\s*([0-9,，\s]+)',  # Match "输出: 1,2" or "输出：1,2"
            r'输出[：:]\s*([0-9]+)',      # Match single digit
            r'[：:]\s*([0-9,，\s]+)',     # Match any colon followed by digits
            r'^([0-9,，\s]+)$',           # Match entire line with only digits and commas
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output.strip())
            if match:
                error_code = match.group(1).strip()
                # Clean and standardize error code
                error_code = self._clean_error_code(error_code)
                if self._validate_error_code(error_code):
                    return error_code
        
        # Method 2: Try direct splitting
        # try:
        #     if ':' in output:
        #         parts = output.split(':')
        #         if len(parts) >= 2:
        #             error_code = parts[1].strip().split('\n')[0].strip()
        #             error_code = self._clean_error_code(error_code)
        #             if self._validate_error_code(error_code):
        #                 return error_code
        # except (IndexError, AttributeError):
        #     pass
        
        return None
    
    def _clean_error_code(self, error_code):
        """
        Clean and standardize error code
        
        Args:
            error_code (str): Raw error code
            
        Returns:
            str: Cleaned error code
        """
        
        # Remove extra spaces and punctuation
        error_code = re.sub(r'[^\d,，]', '', error_code)
        # Replace Chinese commas with English commas
        error_code = error_code.replace('，', ',')
        # Remove repeated commas
        error_code = re.sub(r',+', ',', error_code)
        # Remove leading and trailing commas
        error_code = error_code.strip(',')
        
        return error_code
    
    def _validate_error_code(self, error_code):
        """
        Validate if error code format is correct
        
        Args:
            error_code (str): Error code
            
        Returns:
            bool: Whether it's a valid error code
        """
        
        # Check if only contains digits and commas
        if not re.match(r'^[0-9,]*$', error_code):
            return False
        
        # Check if empty
        if not error_code:
            return False
        
        # Check if each digit is within valid range (0-7)
        try:
            numbers = [int(x.strip()) for x in error_code.split(',') if x.strip()]
            return all(0 <= num <= 7 for num in numbers)
        except ValueError:
            return False
    
    def _add_format_reminder_to_prompt(self, original_prompt):
        """
        Add format reminder to the end of prompt
        
        Args:
            original_prompt (str): Original prompt
            
        Returns:
            str: Prompt with format reminder added
        """
        reminder = """
        
Important reminder: Please output strictly according to the following format, do not add any other text:
Output: [Error Number]

For example:
Output: 0
Output: 1
Output: 1,5

Please check the following medical record now:
"""
        return original_prompt + reminder

    def batch_check_errors(self, datas):
        """
        Batch check data errors
        
        Args:
            datas (list): Data list
        
        Returns:
            list: Check result list
        """
        checked = []
        
        for item in datas:
            result = self.check_data_error(item)
            checked.append(result)
        
        return checked

    @staticmethod
    def extract_symptom_from_output(data_string, spliter=" || "):
        """
        Extract symptom part from formatted data string
        
        Args:
            data_string (str): Formatted data string, e.g., "29岁 || 女 || 成人 || 尿频、尿急、尿痛 || 尿路感染"
            spliter (str): Separator, default is " || "
        
        Returns:
            str: Symptom part content, returns None if format is incorrect
        """
        try:
            parts = data_string.split(spliter)
            if len(parts) >= 4:
                return parts[3]  # The 4th element (index 3) is the symptom part
            else:
                print(f"Data format incorrect, expected at least 4 parts, actually got {len(parts)} parts")
                return None
        except Exception as e:
            print(f"Error extracting symptom part: {e}")
            return None
    



def load_data(data_path, file_names):
    """
    Load data files
    
    Args:
        data_path (str): Data file path
        file_names (list): Data file name list
    
    Returns:
        list: Combined data list
    """
    datas = []
    for file in file_names:
        with open(os.path.join(data_path, file), "rb") as f:
            data = pickle.load(f)
            datas.extend(data)
    return datas


def save_results(results, output_path):
    """
    Save check results to JSON file
    
    Args:
        results (list): Check result list
        output_path (str): Output file path
    """
    with open(output_path, "w", encoding='utf-8') as fp:
        fp.write(json.dumps(results, ensure_ascii=False, indent=4))


def main(data_path="gnn-data/improvement_66_listtodict", 
         files=["dev.pkl"], 
         output_file="check_out_result_dev.json",
         api_key="sk-d1255a437700465a8709fd302d31834b",
         model="qwen-max"):
    """
    Main function
    
    Args:
        data_path (str): Data file path
        files (list): Data file name list
        output_file (str): Output file name
        api_key (str): API key
        model (str): Model name
    """
    # Initialize LLMAPI
    llm_api = LLMAPI(api_key, model)
    
    # Load data
    datas = load_data(data_path, files)
    
    # Batch check errors
    checked_results = llm_api.batch_check_errors(datas)
    
    # Save results
    output_path = os.path.join(data_path, output_file)
    save_results(checked_results, output_path)
    
    print(f"Check completed, results saved to: {output_path}")





if __name__ == "__main__":
    # Configuration parameters
    path = "gnn-data/improvement_66_listtodict"
    files = ["dev.pkl"]
    output_file_json = "check_out_result_dev.json"
    api_key = "sk-d1255a437700465a8709fd302d31834b"
    
    # Run main function
    main(
        data_path=path,
        files=files,
        output_file=output_file_json,
        api_key=api_key
    )
