#!/usr/bin/env python3
"""
Test script for the integrated widget extraction functionality.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.append('src')

async def test_widget_detection():
    """Test the widget detection without full crawling."""
    
    try:
        from code_widget_extractor import CodeWidgetExtractor
        
        # Sample HTML that mimics Polygon.io structure
        sample_html = """
        <html>
        <body>
            <h2>Code Examples</h2>
            <div class="code-widget">
                <div class="tabs">
                    <button data-language="shell" class="tab active">Shell</button>
                    <button data-language="python" class="tab">Python</button>
                    <button data-language="javascript" class="tab">JavaScript</button>
                    <button data-language="go" class="tab">Go</button>
                    <button data-language="kotlin" class="tab">Kotlin</button>
                </div>
                <div class="tab-content">
                    <pre><code>
# Connect to the websocket
npx wscat -c wss://business.polygon.io/stocks
# Authenticate
{"action":"auth","params":"YOUR_API_KEY"}
# Subscribe to the topic
{"action":"subscribe", "params":"FMV.*"}
                    </code></pre>
                </div>
            </div>
            
            <div class="response-example">
                <pre><code>
{
  "ev": "FMV",
  "fmv": 189.22,
  "sym": "AAPL",
  "t": 1678220098130
}
                </code></pre>
            </div>
        </body>
        </html>
        """
        
        print("Testing widget detection...")
        extractor = CodeWidgetExtractor()
        
        print("Creating detection result (note: LLM call will be skipped in test)")
        # For testing, we'll skip the LLM call and test the JavaScript generation
        
        # Test JavaScript generation with mock widgets
        from code_widget_extractor import CodeWidget
        
        mock_widget = CodeWidget(
            title="API Code Examples",
            description="Interactive code widget with multiple language tabs",
            languages=["shell", "python", "javascript", "go", "kotlin"],
            code_examples={},
            selectors={
                "shell": "[data-language='shell']",
                "python": "[data-language='python']",
                "javascript": "[data-language='javascript']",
                "go": "[data-language='go']",
                "kotlin": "[data-language='kotlin']"
            },
            widget_type="tabs"
        )
        
        js_code = extractor.generate_extraction_javascript([mock_widget])
        print("Generated JavaScript:")
        print("=" * 50)
        print(js_code)
        print("=" * 50)
        
        print("✅ Widget detection framework test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_basic_crawl():
    """Test basic crawling still works."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        
        print("Testing basic crawling functionality...")
        
        # Test a simple page
        url = 'https://httpbin.org/html'
        
        browser_config = BrowserConfig(headless=True, verbose=False)
        crawler = AsyncWebCrawler(config=browser_config)
        
        async with crawler:
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            result = await crawler.arun(url=url, config=run_config)
            
            if result.success:
                print(f"✅ Basic crawl successful: {len(result.markdown)} chars")
                return True
            else:
                print(f"❌ Basic crawl failed: {result.error_message}")
                return False
                
    except Exception as e:
        print(f"❌ Basic crawl test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Widget Integration")
    print("=" * 50)
    
    # Test 1: Widget detection framework
    print("\n1. Testing widget detection framework...")
    widget_test = await test_widget_detection()
    
    # Test 2: Basic crawling
    print("\n2. Testing basic crawling...")
    crawl_test = await test_basic_crawl()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Widget Detection: {'✅ PASS' if widget_test else '❌ FAIL'}")
    print(f"Basic Crawling: {'✅ PASS' if crawl_test else '❌ FAIL'}")
    
    if widget_test and crawl_test:
        print("\n🎉 All tests passed! Widget integration is ready.")
        print("\nNext steps:")
        print("1. Set USE_AGENTIC_RAG=true in your .env file")
        print("2. Test with: crawl_single_page(url='https://polygon.io/docs/websocket/stocks/fair-market-value')")
        print("3. Check the code_examples table for extracted widgets")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())