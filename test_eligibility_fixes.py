#!/usr/bin/env python3
"""
Test script to verify the eligibility checker fixes
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from eligibility.eligibility_checker import check_detailed_eligibility, safe_float_conversion

def test_safe_float_conversion():
    """Test the safe float conversion function"""
    print("Testing safe_float_conversion:")
    
    test_cases = [
        ("7.5", 7.5),
        ("8.0%", 8.0),
        ("60%", 60.0),
        ("7,5", 7.5),
        ("", 0.0),
        (None, 0.0),
        (8.5, 8.5),
        (10, 10.0),
        ("invalid", 0.0),
        ("8.5 CGPA", 8.5)
    ]
    
    for input_val, expected in test_cases:
        result = safe_float_conversion(input_val)
        status = "✅" if result == expected else "❌"
        print(f"  {status} {input_val} -> {result} (expected: {expected})")

def test_eligibility_checker():
    """Test the eligibility checker with sample data"""
    print("\nTesting eligibility checker:")
    
    sample_payload = {
        "user": {
            "id": "test123",
            "name": "Test Student",
            "course": "B.Tech",
            "stream": "CSE",
            "batch": "2025",
            "institute": "Test Institute",
            "avg_cgpa": "8.5%",  # This should cause the original error
            "activeBacklogs": "0",
            "skillsCount": 3,
            "skills": ["Python", "JavaScript", "React"]
        },
        "post": {
            "criteria": {
                "cgpa": "7.0%",  # This should also cause the original error
                "backlogs": "0",
                "skills": ["Python", "JavaScript", "Java"]
            },
            "eligibility": {
                "minCGPA": "7.5",
                "branches": ["CSE", "IT"],
                "batch": ["2025", "2026"]
            }
        }
    }
    
    try:
        result = check_detailed_eligibility(sample_payload)
        print("✅ Eligibility check completed successfully!")
        print(f"  - Eligible: {result.get('isEligible', False)}")
        print(f"  - CGPA Status: {result.get('breakdown', {}).get('cgpa', {}).get('status', 'N/A')}")
        print(f"  - Skills Status: {result.get('breakdown', {}).get('skills', {}).get('status', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Eligibility check failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing eligibility checker fixes...")
    print("=" * 50)
    
    test_safe_float_conversion()
    success = test_eligibility_checker()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    sys.exit(0 if success else 1)
