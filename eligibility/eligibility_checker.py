import json
from typing import Dict, Any, List
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class EligibilityPayload(BaseModel):
    user: Dict[str, Any]
    post: Dict[str, Any]

def match_user_skills_with_required(user_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
    """
    Use Groq AI to match user skills with required skills.
    """
    if not required_skills:
        return {
            "status": "pass",
            "message": "No specific skills required",
            "matchedSkills": [],
            "missingSkills": []
        }
    
    if not user_skills:
        return {
            "status": "fail",
            "message": f"None of the {len(required_skills)} required skills matched",
            "matchedSkills": [],
            "missingSkills": required_skills
        }
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    user_skills_str = ", ".join(user_skills)
    required_skills_str = ", ".join(required_skills)
    
    prompt = f"""
    You are a skills matcher for recruitment eligibility.
    
    USER SKILLS: {user_skills_str}
    REQUIRED SKILLS: {required_skills_str}
    
    For each required skill, determine if the user has that skill or a closely related skill.
    Consider:
    - Exact matches
    - Similar technologies or frameworks
    - Related domains (e.g., CS includes Computer Science, CSE)
    - Abbreviations and full forms (e.g., IT = Information Technology)
    - Note: IT, ECE, EE might refer to academic branches, not necessarily skills
    
    Return ONLY a JSON object with matched and missing skills:
    {{
        "matchedSkills": ["skill1", "skill2"],
        "missingSkills": ["skill3", "skill4"]
    }}
    
    Be thorough but fair in your assessment.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated to working model
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        content = chat_completion.choices[0].message.content.strip()
        print("Skills Matching AI Response:", content)
        
        # Try to extract JSON from the response
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                skills_result = json.loads(json_str)
                
                matched_skills = skills_result.get("matchedSkills", [])
                missing_skills = skills_result.get("missingSkills", [])
                
                # Determine status
                if len(matched_skills) == len(required_skills):
                    status = "pass"
                    message = f"All {len(required_skills)} required skills matched"
                elif len(matched_skills) > 0:
                    status = "partial"
                    message = f"{len(matched_skills)} out of {len(required_skills)} required skills matched"
                else:
                    status = "fail"
                    message = f"None of the {len(required_skills)} required skills matched"
                
                return {
                    "status": status,
                    "message": message,
                    "matchedSkills": matched_skills,
                    "missingSkills": missing_skills
                }
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing skills matching response: {e}")
            # Fallback: basic string matching
            return fallback_skills_matching(user_skills, required_skills)
            
    except Exception as e:
        print(f"Error in skills matching: {e}")
        # Fallback: basic string matching
        return fallback_skills_matching(user_skills, required_skills)

def fallback_skills_matching(user_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
    """
    Fallback skills matching using basic string comparison.
    """
    matched_skills = []
    missing_skills = []
    
    # Create skill mappings for better matching
    skill_mappings = {
        'python': ['python', 'py'],
        'javascript': ['javascript', 'js', 'node.js', 'nodejs'],
        'java': ['java'],
        'it': ['information technology', 'it', 'computer science', 'cs', 'cse'],
        'ece': ['electronics and communication', 'ece', 'electronics'],
        'ee': ['electrical engineering', 'ee', 'electrical'],
        'me': ['mechanical engineering', 'me', 'mechanical'],
        'ce': ['civil engineering', 'ce', 'civil']
    }
    
    # Convert user skills to lowercase for comparison
    user_skills_lower = [skill.lower().strip() for skill in user_skills]
    
    for req_skill in required_skills:
        req_skill_lower = req_skill.lower().strip()
        is_matched = False
        
        # Direct match
        if req_skill_lower in user_skills_lower:
            matched_skills.append(req_skill)
            is_matched = True
        else:
            # Check mappings
            if req_skill_lower in skill_mappings:
                for user_skill in user_skills_lower:
                    if user_skill in skill_mappings[req_skill_lower]:
                        matched_skills.append(req_skill)
                        is_matched = True
                        break
            
            # Check reverse mappings
            if not is_matched:
                for user_skill in user_skills_lower:
                    if user_skill in skill_mappings:
                        if req_skill_lower in skill_mappings[user_skill]:
                            matched_skills.append(req_skill)
                            is_matched = True
                            break
            
            # Partial string matching
            if not is_matched:
                for user_skill in user_skills_lower:
                    if req_skill_lower in user_skill or user_skill in req_skill_lower:
                        matched_skills.append(req_skill)
                        is_matched = True
                        break
        
        if not is_matched:
            missing_skills.append(req_skill)
    
    # Determine status
    if len(matched_skills) == len(required_skills):
        status = "pass"
        message = f"All {len(required_skills)} required skills matched"
    elif len(matched_skills) > 0:
        status = "partial"
        message = f"{len(matched_skills)} out of {len(required_skills)} required skills matched"
    else:
        status = "fail"
        message = f"None of the {len(required_skills)} required skills matched"
    
    return {
        "status": status,
        "message": message,
        "matchedSkills": matched_skills,
        "missingSkills": missing_skills
    }

def check_eligibility_with_ai(user_data: Dict[str, Any], eligibility_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Groq AI to perform comprehensive eligibility checking.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""
    You are an eligibility checker for campus recruitment.
    
    USER PROFILE:
    - Name: {user_data.get('name', 'N/A')}
    - Course: {user_data.get('course', 'N/A')}
    - Stream: {user_data.get('stream', 'N/A')}
    - Batch: {user_data.get('batch', 'N/A')}
    - CGPA: {user_data.get('avg_cgpa', 0.0)}
    - Active Backlogs: {user_data.get('activeBacklogs', 0)}
    
    ELIGIBILITY CRITERIA:
    - Minimum CGPA: {eligibility_criteria.get('minCGPA', 0.0)}
    - Eligible Branches: {eligibility_criteria.get('branches', [])}
    - Eligible Batches: {eligibility_criteria.get('batch', [])}
    - Maximum Backlogs Allowed: {eligibility_criteria.get('backlogs', 0)}
    
    Check each criterion and provide detailed analysis:
    1. CGPA: Compare user CGPA with minimum requirement
    2. Branch/Stream: Check if user's stream is in eligible branches (consider "All" means all branches)
    3. Batch: Check if user's batch matches eligible batches
    4. Backlogs: Check if user's backlogs are within allowed limit
    
    Return STRICTLY in this JSON format:
    {{
        "cgpa": {{
            "status": "pass" or "fail",
            "message": "detailed explanation"
        }},
        "course": {{
            "status": "pass" or "fail", 
            "message": "detailed explanation"
        }},
        "batch": {{
            "status": "pass" or "fail",
            "message": "detailed explanation" 
        }},
        "backlogs": {{
            "status": "pass" or "fail",
            "message": "detailed explanation"
        }},
        "overallEligible": true or false
    }}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated to working model
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        content = chat_completion.choices[0].message.content.strip()
        print("Eligibility AI Response:", content)
        
        # Try to extract JSON from the response
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing AI response: {e}")
            # Fallback to manual checking
            return manual_eligibility_check(user_data, eligibility_criteria)
            
    except Exception as e:
        print(f"Error in AI eligibility checking: {e}")
        # Fallback to manual checking
        return manual_eligibility_check(user_data, eligibility_criteria)

def manual_eligibility_check(user_data: Dict[str, Any], eligibility_criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback manual eligibility checking.
    """
    avg_cgpa = user_data.get('avg_cgpa', 0.0)
    stream = user_data.get('stream', '')
    batch = user_data.get('batch', '')
    active_backlogs = user_data.get('activeBacklogs', 0)
    
    min_cgpa = float(eligibility_criteria.get('minCGPA', 0.0))
    eligible_branches = eligibility_criteria.get('branches', [])
    eligible_batches = eligibility_criteria.get('batch', [])
    max_backlogs = eligibility_criteria.get('backlogs', 0)
    
    # Check CGPA
    cgpa_status = "pass" if avg_cgpa >= min_cgpa else "fail"
    cgpa_message = f"Your CGPA ({avg_cgpa}) {'meets' if cgpa_status == 'pass' else 'is below'} the minimum requirement ({min_cgpa})"
    
    # Check course/branch
    course_status = "pass" if ("All" in eligible_branches or stream.upper() in [b.upper() for b in eligible_branches]) else "fail"
    course_message = f"Your course ({stream}) {'is eligible' if course_status == 'pass' else 'is not in the eligible branches: ' + ', '.join(eligible_branches)}"
    
    # Check batch
    batch_status = "pass" if str(batch) in [str(b) for b in eligible_batches] else "fail"
    batch_message = f"Your batch ({batch}) {'is eligible' if batch_status == 'pass' else 'is not eligible. Eligible batches: ' + ', '.join(map(str, eligible_batches))}"
    
    # Check backlogs
    backlogs_status = "pass" if active_backlogs <= max_backlogs else "fail"
    backlogs_message = f"You have {active_backlogs} active backlog(s), {'which meets' if backlogs_status == 'pass' else 'but maximum'} the requirement (≤{max_backlogs})"
    
    overall_eligible = all([cgpa_status == "pass", course_status == "pass", batch_status == "pass", backlogs_status == "pass"])
    
    return {
        "cgpa": {"status": cgpa_status, "message": cgpa_message},
        "course": {"status": course_status, "message": course_message},
        "batch": {"status": batch_status, "message": batch_message},
        "backlogs": {"status": backlogs_status, "message": backlogs_message},
        "overallEligible": overall_eligible
    }

def check_detailed_eligibility(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check eligibility based on the provided payload and return detailed breakdown using AI.
    """
    user = payload["user"]
    post = payload["post"]
    criteria = post.get("criteria", {})
    eligibility = post.get("eligibility", {})
    
    # Extract user details
    user_id = user.get("id", "")
    name = user.get("name", "")
    course = user.get("course", "")
    stream = user.get("stream", "")
    batch = user.get("batch", "")
    institute = user.get("institute", "")
    avg_cgpa = user.get("avg_cgpa", 0.0)
    active_backlogs = user.get("activeBacklogs", 0)
    skills_count = user.get("skillsCount", 0)
    user_skills = user.get("skills", [])
    
    # Extract skill names from user skills (handle both string and object formats)
    user_skill_names = []
    for skill in user_skills:
        if isinstance(skill, str):
            user_skill_names.append(skill)
        elif isinstance(skill, dict) and "name" in skill:
            user_skill_names.append(skill["name"])
    
    # Combine eligibility criteria from both sources
    combined_eligibility = {
        "minCGPA": eligibility.get("minCGPA", criteria.get("cgpa", 0.0)),
        "branches": eligibility.get("branches", []),
        "batch": eligibility.get("batch", []),
        "backlogs": criteria.get("backlogs", 0)
    }
    
    # Required skills from criteria
    required_skills = criteria.get("skills", [])
    
    # Use AI to check basic eligibility criteria
    ai_eligibility = check_eligibility_with_ai(user, combined_eligibility)
    
    # Use AI/skills matcher to check skills
    skills_result = match_user_skills_with_required(user_skill_names, required_skills)
    
    # Build breakdown from AI results
    breakdown = {
        "cgpa": ai_eligibility.get("cgpa", {"status": "fail", "message": "CGPA check failed"}),
        "backlogs": ai_eligibility.get("backlogs", {"status": "fail", "message": "Backlogs check failed"}),
        "course": ai_eligibility.get("course", {"status": "fail", "message": "Course check failed"}),
        "batch": ai_eligibility.get("batch", {"status": "fail", "message": "Batch check failed"}),
        "skills": skills_result
    }
    
    # Determine overall eligibility
    basic_eligibility = ai_eligibility.get("overallEligible", False)
    
    # FIXED: Candidate is eligible if they meet basic criteria, regardless of skills
    # Skills are additional information, not blocking criteria
    is_eligible = basic_eligibility
    
    # Generate recommendations
    recommendations = []
    
    if breakdown["cgpa"]["status"] == "fail":
        min_cgpa = combined_eligibility.get("minCGPA", 0.0)
        recommendations.append(f"Improve your CGPA to at least {min_cgpa}")
    
    if breakdown["backlogs"]["status"] == "fail":
        recommendations.append("Clear your active backlogs before applying")
    
    if breakdown["course"]["status"] == "fail":
        eligible_branches = combined_eligibility.get("branches", [])
        recommendations.append(f"This opportunity is only for {', '.join(eligible_branches)} branches")
    
    if breakdown["batch"]["status"] == "fail":
        eligible_batches = combined_eligibility.get("batch", [])
        recommendations.append(f"This opportunity is only for {', '.join(map(str, eligible_batches))} batch")
    
    # Skills recommendations (informational, not blocking)
    if skills_result["status"] == "fail" and required_skills:
        missing_skills = skills_result.get("missingSkills", [])
        recommendations.append(f"Consider developing skills in: {', '.join(missing_skills)} to strengthen your profile")
    elif skills_result["status"] == "partial":
        missing_skills = skills_result.get("missingSkills", [])
        if missing_skills:
            recommendations.append(f"Consider developing additional skills in: {', '.join(missing_skills)}")
    
    # Add general recommendations
    if is_eligible:
        recommendations.append("You are eligible! Prepare well for the selection process")
        recommendations.append("Review the job description and company information thoroughly")
        if skills_result["status"] == "pass":
            recommendations.append("Your skills align well with the requirements")
        elif skills_result["status"] == "partial":
            recommendations.append("You have some of the desired skills - highlight them in your application")
    else:
        recommendations.append("Focus on meeting the basic eligibility criteria before applying")
    
    # Build response
    response = {
        "id": user_id,
        "name": name,
        "course": course,
        "stream": stream,
        "batch": batch,
        "institute": institute,
        "avg_cgpa": avg_cgpa,
        "activeBacklogs": active_backlogs,
        "skillsCount": skills_count,
        "skills": user_skills,
        "isEligible": is_eligible,
        "breakdown": breakdown,
        "recommendations": recommendations
    }
    
    return response