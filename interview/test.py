import requests
from bs4 import BeautifulSoup
import json
import time
import urllib.parse
from typing import List, Dict, Any
import re

class InterviewQuestionsScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\?\.\,\-\(\)\:]', '', text)
        return text
    
    def scrape_ambitionbox(self, company_name: str, role: str) -> List[str]:
        """Scrape interview questions from AmbitionBox"""
        questions = []
        try:
            # Format company name for URL
            company_slug = company_name.lower().replace(' ', '-').replace('.', '').replace(',', '')
            
            # Multiple URL patterns to try
            url_patterns = [
                f"https://www.ambitionbox.com/interviews/{company_name}-interview-questions",
            ]
            
            for url in url_patterns:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for interview questions with various selectors
                        question_selectors = [
                            '.interview-question',
                            '.question-text',
                            '[class*="question"]',
                            '.interview-content p',
                            '.review-text'
                        ]
                        
                        for selector in question_selectors:
                            elements = soup.select(selector)
                            for element in elements:
                                text = self.clean_text(element.get_text())
                                if text and '?' in text and len(text) > 10:
                                    questions.append(text)
                        
                        if questions:
                            break
                            
                except Exception as e:
                    print(f"Error with AmbitionBox URL {url}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"AmbitionBox scraping error: {str(e)}")
        
        return list(set(questions))[:10]  # Remove duplicates and limit to 10
    
    def scrape_glassdoor(self, company_name: str, role: str) -> List[str]:
        """Scrape interview questions from Glassdoor India"""
        questions = []
        try:
            # Search for company interview questions
            search_query = f"{company_name} {role} interview questions"
            encoded_query = urllib.parse.quote_plus(search_query)
            
            url_patterns = [
                f"https://www.glassdoor.co.in/Interview/{company_name.replace(' ', '-')}-Interview-Questions-E*.htm",
                f"https://www.glassdoor.co.in/Interview/interview-questions.htm?q={encoded_query}",
            ]
            
            for url in url_patterns:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for interview questions
                        question_selectors = [
                            '[data-test="interview-question"]',
                            '.interviewQuestion',
                            '.interview-question-text',
                            '[class*="InterviewQuestion"]',
                            '.questionText'
                        ]
                        
                        for selector in question_selectors:
                            elements = soup.select(selector)
                            for element in elements:
                                text = self.clean_text(element.get_text())
                                if text and '?' in text and len(text) > 10:
                                    questions.append(text)
                        
                        if questions:
                            break
                            
                except Exception as e:
                    print(f"Error with Glassdoor URL {url}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Glassdoor scraping error: {str(e)}")
        
        return list(set(questions))[:10]
    
    def scrape_geeksforgeeks(self, company_name: str, role: str) -> List[str]:
        """Scrape interview questions from GeeksforGeeks"""
        questions = []
        try:
            # Search for company-specific interview questions
            search_terms = [
                f"{company_name.lower()}-interview-questions",
                f"{company_name.lower()}-interview-experience",
                f"{role.lower()}-interview-questions-{company_name.lower()}"
            ]
            
            for search_term in search_terms:
                try:
                    url = f"https://www.geeksforgeeks.org/{search_term}/"
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Look for interview questions
                        question_selectors = [
                            '.content p',
                            'li',
                            '.interview-question',
                            'h3',
                            'h4'
                        ]
                        
                        for selector in question_selectors:
                            elements = soup.select(selector)
                            for element in elements:
                                text = self.clean_text(element.get_text())
                                if text and '?' in text and len(text) > 10:
                                    questions.append(text)
                        
                        if questions:
                            break
                            
                except Exception as e:
                    print(f"Error with GeeksforGeeks URL {url}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"GeeksforGeeks scraping error: {str(e)}")
        
        return list(set(questions))[:10]
    
    def get_interview_questions(self, company_name: str, role: str) -> Dict[str, Any]:
        """
        Main function to get interview questions from all three websites
        
        Args:
            company_name (str): Name of the company (e.g., "Google", "Microsoft")
            role (str): Job role (e.g., "Software Engineer", "Data Scientist")
            
        Returns:
            Dict: JSON format with questions from all sources
        """
        print(f"Fetching interview questions for {role} at {company_name}...")
        
        # Initialize results
        results = {
            "company": company_name,
            "role": role,
            "sources": {
                "ambitionbox": [],
                "glassdoor": [],
                "geeksforgeeks": []
            },
            "total_questions": 0,
            "all_questions": []
        }
        
        # Scrape from AmbitionBox
        print("Scraping AmbitionBox...")
        ambitionbox_questions = self.scrape_ambitionbox(company_name, role)
        results["sources"]["ambitionbox"] = ambitionbox_questions
        time.sleep(1)  # Be respectful to servers
        
        # Scrape from Glassdoor
        print("Scraping Glassdoor...")
        glassdoor_questions = self.scrape_glassdoor(company_name, role)
        results["sources"]["glassdoor"] = glassdoor_questions
        time.sleep(1)
        
        # Scrape from GeeksforGeeks
        print("Scraping GeeksforGeeks...")
        geeksforgeeks_questions = self.scrape_geeksforgeeks(company_name, role)
        results["sources"]["geeksforgeeks"] = geeksforgeeks_questions
        time.sleep(1)
        
        # Combine all questions and remove duplicates
        all_questions = []
        for source_questions in results["sources"].values():
            all_questions.extend(source_questions)
        
        # Remove duplicates while preserving order
        unique_questions = []
        seen = set()
        for q in all_questions:
            q_lower = q.lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_questions.append(q)
        
        results["all_questions"] = unique_questions
        results["total_questions"] = len(unique_questions)
        
        return results

# Main function to use
def get_interview_questions(company_name: str, role: str) -> str:
    """
    Get interview questions for a specific company and role
    
    Args:
        company_name (str): Name of the company
        role (str): Job role
        
    Returns:
        str: JSON string with interview questions
    """
    scraper = InterviewQuestionsScraper()
    results = scraper.get_interview_questions(company_name, role)
    return json.dumps(results, indent=2)

# Example usage
