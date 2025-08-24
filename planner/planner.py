import os
import json
import re
from groq import Groq
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

def extract_urls_from_search_result(search_text):
    """Extract URLs from DuckDuckGo search result text"""
    # Pattern to match URLs in the search results
    url_pattern = r'https?://[^\s\])]+'
    urls = re.findall(url_pattern, search_text)
    
    # Clean up URLs and remove duplicates
    clean_urls = []
    seen_domains = set()
    
    for url in urls:
        # Clean trailing punctuation
        url = re.sub(r'[.,;!?\])}]+$', '', url)
        
        # Extract domain to avoid duplicates from same site
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            if domain not in seen_domains and len(url) > 10:  # Basic validity check
                clean_urls.append(url)
                seen_domains.add(domain)
                
    return clean_urls

def generate_resource_title(module_title, url, resource_index):
    """Generate a meaningful title for the resource based on URL and module"""
    # Extract site name from URL
    site_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    site_name = site_match.group(1) if site_match else "Resource"
    
    # Common educational sites
    educational_sites = {
        'youtube.com': 'Video Tutorial',
        'medium.com': 'Article',
        'dev.to': 'Tutorial',
        'stackoverflow.com': 'Q&A Resource',
        'github.com': 'Code Repository',
        'docs.python.org': 'Documentation',
        'developer.mozilla.org': 'Documentation',
        'w3schools.com': 'Tutorial',
        'geeksforgeeks.org': 'Tutorial',
        'leetcode.com': 'Practice Problems',
        'hackerrank.com': 'Practice Platform',
        'coursera.org': 'Course',
        'udemy.com': 'Course',
        'edx.org': 'Course'
    }
    
    # Determine resource type based on URL
    resource_type = "Resource"
    for site, type_name in educational_sites.items():
        if site in url.lower():
            resource_type = type_name
            break
    
    # Create title based on module and resource type
    if resource_index == 0:
        return f"{module_title} - {resource_type}"
    else:
        return f"{module_title} - {resource_type} {resource_index + 1}"

def determine_resource_type(url):
    """Determine resource type based on URL"""
    url_lower = url.lower()
    
    if any(site in url_lower for site in ['youtube.com', 'vimeo.com']):
        return 'video'
    elif any(site in url_lower for site in ['github.com', 'gitlab.com']):
        return 'repository'
    elif any(site in url_lower for site in ['leetcode.com', 'hackerrank.com', 'codewars.com']):
        return 'practice'
    elif any(site in url_lower for site in ['docs.', 'documentation']):
        return 'documentation'
    elif any(site in url_lower for site in ['coursera.org', 'udemy.com', 'edx.org']):
        return 'course'
    else:
        return 'article'

def generate_plan(payload: dict):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model_name = "llama-3.1-8b-instant"
    search = DuckDuckGoSearchRun()

    # Step 1: Ask Groq to build the study plan
    base_prompt = f"""
    Based on this candidate payload:

    {payload}

    Create ONLY a JSON preparation plan that follows this EXACT structure:

    {{
      "id": "plan-1",
      "title": "{payload['company']} {payload['role']} Preparation Plan",
      "estimatedTime": "X-Y weeks",
      "difficulty": "Easy/Medium/Hard",
      "modules": [
        {{
          "id": "module-1",
          "title": "Data Structures & Algorithms Fundamentals",
          "duration": "X weeks",
          "description": "Brief description of what this module covers",
          "resources": []
        }},
        {{
          "id": "module-2",
          "title": "System Design Basics",
          "type": "reading",
          "duration": "X week",
          "description": "Brief description of what this module covers",
          "resources": []
        }},
        {{
          "id": "module-3",
          "title": "Mock Interviews",
          "type": "practice",
          "duration": "X week",
          "description": "Brief description of what this module covers",
          "resources": []
        }}
      ]
    }}

    Requirements:
    - Generate 3-4 modules based on the role and company requirements
    - Include "type" field for modules where appropriate (reading, practice, coding, etc.)
    - Make sure estimatedTime is in "X-Y weeks" format
    - Set appropriate difficulty level based on role and candidate skills
    - Focus on skills mentioned in the criteria: {payload['criteria']['skills']}
    - Return ONLY valid JSON, no other text
    - Do not include any markdown formatting or code blocks
    """

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": base_prompt}],
        temperature=0.3
    )
    
    plan_text = resp.choices[0].message.content.strip()
    
    # Clean up potential markdown formatting
    if plan_text.startswith('```json'):
        plan_text = plan_text[7:]
    if plan_text.endswith('```'):
        plan_text = plan_text[:-3]
    plan_text = plan_text.strip()
    
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {plan_text}")
        return {"error": "Failed to parse JSON response", "raw_response": plan_text}

    # Step 2: Enrich modules with resources via web search
    for i, module in enumerate(plan.get("modules", []), start=1):
        # Create more specific search queries for better results
        role_keywords = payload['role'].lower().replace(' ', ' ')
        company = payload['company']
        
        search_queries = [
            f"{module['title']} tutorial {role_keywords}",
            f"{module['title']} {company} interview",
            f"learn {module['title']} programming"
        ]
        
        resources = []
        all_urls = []
        
        # Collect URLs from all search queries
        for query in search_queries:
            try:
                print(f"Searching for: {query}")
                search_result = search.run(query)
                urls = extract_urls_from_search_result(search_result)
                all_urls.extend(urls)
            except Exception as e:
                print(f"Search error for module {i}, query '{query}': {e}")
        
        # Remove duplicates and take best URLs
        unique_urls = list(dict.fromkeys(all_urls))  # Preserves order while removing duplicates
        
        # Create resources from URLs (limit to 3-4 per module)
        for j, url in enumerate(unique_urls[:4]):
            try:
                title = generate_resource_title(module['title'], url, j)
                resource_type = determine_resource_type(url)
                
                resources.append({
                    "title": title,
                    "url": url,
                    "type": resource_type
                })
            except Exception as e:
                print(f"Error creating resource for URL {url}: {e}")
        
        # Fallback: if no URLs found, create generic resources
        if not resources:
            fallback_resources = [
                {
                    "title": f"{module['title']} - Tutorial",
                    "url": f"https://www.google.com/search?q={module['title'].replace(' ', '+').lower()}+tutorial",
                    "type": "article"
                },
                {
                    "title": f"{module['title']} - Video Guide",
                    "url": f"https://www.youtube.com/results?search_query={module['title'].replace(' ', '+').lower()}",
                    "type": "video"
                }
            ]
            resources = fallback_resources[:2]
        
        module["resources"] = resources
        # Ensure IDs are consistent
        module["id"] = f"module-{i}"
        
        print(f"Module {i} ({module['title']}): Found {len(resources)} resources")

    # Ensure plan consistency
    plan["id"] = "plan-1"
    plan["title"] = f"{payload['company']} {payload['role']} Preparation Plan"

    return plan


# ---- Static input ----
payload = {
  "course": "Bachelor of Technology",
  "stream": "Computer Science Engineering",
  "avg_cgpa": 8.2,
  "activeBacklogs": 1,
  "skills": [
    { "name": "JavaScript", "level": "Advanced" },
    { "name": "React", "level": "Intermediate" },
    { "name": "Python", "level": "Advanced" },
    { "name": "Node.js", "level": "Intermediate" },
    { "name": "SQL", "level": "Beginner" }
  ],
  "company": "Google",
  "role": "Software Development Engineer Intern",
  "ctc": "₹80,000/month + Benefits",
  "applicationProcess": [
    "Submit application through campus placement portal",
    "Online coding assessment (2 hours)",
    "Technical interview rounds (2-3 rounds)",
    "HR interview and background verification",
    "Offer letter and documentation"
  ],
  "criteria": {
    "skills": ["JavaScript","Python","Data Structures","Algorithms","System Design"],
    "courses": ["Computer Science","Information Technology","Electronics"]
  }
}

if __name__ == "__main__":
    plan = generate_plan(payload)
    print(json.dumps(plan, indent=2))