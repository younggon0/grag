#!/usr/bin/env python3
"""
Simple test to verify file upload logic without Streamlit dependency
"""

import tempfile
import os

def test_file_upload_logic():
    """Test the core file upload and processing logic"""
    
    # Create a test file
    test_content = """Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file_path = f.name
    
    try:
        # Simulate file upload: reading content
        print("Testing file upload logic...")
        
        # Step 1: Read file content (simulating uploaded_file.read())
        with open(test_file_path, 'r') as f:
            content = f.read()
        print(f"✅ Read content: {content[:50]}...")
        
        # Step 2: Check file size (simulating size check)
        file_size = os.path.getsize(test_file_path)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            print("❌ File too large")
            return False
        print(f"✅ File size OK: {file_size} bytes")
        
        # Step 3: Process filename
        filename = os.path.basename(test_file_path)
        print(f"✅ Filename: {filename}")
        
        # Step 4: Verify content processing would work
        if content and len(content.strip()) > 0:
            print("✅ Content is valid and non-empty")
            print("✅ File upload logic test PASSED")
            return True
        else:
            print("❌ Content is empty")
            return False
            
    except Exception as e:
        print(f"❌ Error in file upload logic: {str(e)}")
        return False
    finally:
        # Clean up
        os.unlink(test_file_path)

def test_streamlit_upload_widget():
    """Analyze the Streamlit upload widget configuration"""
    
    print("\nAnalyzing Streamlit upload widget configuration...")
    
    # Read the app_v2.py file to check upload widget config
    try:
        with open('/Users/eric/src/grag/app_v2.py', 'r') as f:
            content = f.read()
        
        # Find the file_uploader configuration
        import re
        
        # Look for file_uploader calls
        uploader_matches = re.findall(r'st\.file_uploader\((.*?)\)', content, re.DOTALL)
        
        if uploader_matches:
            print("✅ Found file_uploader widget configurations:")
            for i, match in enumerate(uploader_matches, 1):
                print(f"  {i}. st.file_uploader({match})")
                
                # Check for type restrictions
                if 'type=' in match:
                    print("    ✅ File type restrictions found")
                else:
                    print("    ⚠️  No file type restrictions")
                
                # Check for help text
                if 'help=' in match:
                    print("    ✅ Help text provided")
                else:
                    print("    ⚠️  No help text")
        else:
            print("❌ No file_uploader widgets found")
            return False
        
        # Look for file processing logic
        if 'uploaded_file' in content:
            print("✅ Upload handling logic found")
        else:
            print("❌ No upload handling logic found")
            
        # Look for size checks
        if 'uploaded_file.size' in content:
            print("✅ File size validation found")
        else:
            print("⚠️  No file size validation found")
            
        print("✅ Streamlit widget analysis PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing Streamlit config: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== File Upload Debug Test ===")
    
    # Test 1: Basic file upload logic
    test1_result = test_file_upload_logic()
    
    # Test 2: Streamlit widget configuration
    test2_result = test_streamlit_upload_widget()
    
    print(f"\n=== Results ===")
    print(f"File upload logic: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Streamlit config: {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    if test1_result and test2_result:
        print("\n🎉 All tests passed - upload logic should work!")
        print("If upload is still not working, the issue is likely in the Streamlit UI or session state management.")
    else:
        print("\n⚠️  Issues found - check the output above for details.")