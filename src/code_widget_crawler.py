"""
Dedicated Code Widget Crawler

A specialized tool for extracting code examples from interactive widgets on documentation sites.
This tool focuses specifically on clicking through language tabs and extracting all code variants.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from utils import extract_code_blocks, generate_code_example_summary, add_code_examples_to_supabase, get_supabase_client, update_source_info, extract_source_summary
from urllib.parse import urlparse
import concurrent.futures

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)


@dataclass
class CodeExtractionResult:
    """Result of code extraction from a specific language tab."""
    language: str
    code: str
    length: int
    success: bool
    error: Optional[str] = None


@dataclass
class WidgetExtractionSummary:
    """Summary of all code extractions from a page."""
    url: str
    total_languages_attempted: int
    successful_extractions: int
    failed_extractions: int
    results: List[CodeExtractionResult]
    total_code_length: int


class CodeWidgetCrawler:
    """
    Specialized crawler for extracting code examples from interactive widgets.
    """
    
    def __init__(self):
        self.browser_config = BrowserConfig(headless=True, verbose=False)
        
        # Common programming languages to look for
        self.target_languages = [
            'shell', 'bash', 'sh',
            'python', 'py', 
            'javascript', 'js',
            'go', 'golang',
            'kotlin', 'kt',
            'java',
            'php',
            'ruby', 'rb',
            'rust', 'rs',
            'typescript', 'ts',
            'curl',
            'powershell', 'ps1',
            'cmd'
        ]
    
    def _generate_language_specific_js(self, target_language: str) -> str:
        """
        Generate JavaScript to click a specific language tab and extract its code.
        """
        return f"""
// Code Widget Crawler - Extract {target_language} code
console.log('Starting {target_language} code extraction...');

// Wait for page to load
await new Promise(r => setTimeout(r, 1000));

// Look for {target_language} tab/button
const targetLanguage = '{target_language.lower()}';
let foundTab = null;
let extractedCode = null;

// Strategy 1: Look for exact language match in buttons/tabs
const languageButtons = document.querySelectorAll(
    'button[data-language], button[aria-controls*="code"], button[role="tab"], .tab, .language-tab, .code-tab, [data-tab]'
);

console.log(`Found ${{languageButtons.length}} potential language buttons`);

for (let button of languageButtons) {{
    const text = (button.textContent || button.innerText || '').trim().toLowerCase();
    const dataLang = (button.getAttribute('data-language') || '').toLowerCase();
    const ariaControls = (button.getAttribute('aria-controls') || '').toLowerCase();
    
    // Check for exact or partial match
    if (text === targetLanguage || 
        dataLang === targetLanguage ||
        text.includes(targetLanguage) ||
        dataLang.includes(targetLanguage) ||
        ariaControls.includes(targetLanguage)) {{
        
        console.log(`Found {target_language} tab: ${{text}} (${{dataLang}})`);
        foundTab = button;
        break;
    }}
}}

// Strategy 2: Look for buttons with language text content
if (!foundTab) {{
    const allClickable = document.querySelectorAll('button, [role="button"], .btn, .clickable, [onclick]');
    for (let el of allClickable) {{
        const text = (el.textContent || el.innerText || '').trim().toLowerCase();
        if (text === targetLanguage || 
            (text.length <= 15 && text.includes(targetLanguage))) {{
            console.log(`Found {target_language} element via text: ${{text}}`);
            foundTab = el;
            break;
        }}
    }}
}}

if (foundTab) {{
    console.log('Clicking {target_language} tab...');
    try {{
        foundTab.click();
        await new Promise(r => setTimeout(r, 1500)); // Wait for content to load
        
        // Extract code content
        const codeBlocks = document.querySelectorAll(
            'pre code, .highlight code, .code-content code, [class*="code"] code, .code-example code, code'
        );
        
        let longestCode = '';
        codeBlocks.forEach(block => {{
            const code = block.textContent || block.innerText || '';
            if (code.length > longestCode.length && code.length > 30) {{
                longestCode = code;
            }}
        }});
        
        if (longestCode) {{
            extractedCode = longestCode;
            console.log(`Extracted ${{longestCode.length}} characters of {target_language} code`);
        }} else {{
            console.log('No code content found after clicking {target_language} tab');
        }}
        
    }} catch (e) {{
        console.log(`Error clicking {target_language} tab: ${{e}}`);
    }}
}} else {{
    console.log('No {target_language} tab found');
}}

// Store result in window for retrieval
window.codeExtractionResult = {{
    language: '{target_language}',
    found: !!foundTab,
    code: extractedCode,
    length: extractedCode ? extractedCode.length : 0
}};

console.log('{target_language} extraction completed');
"""

    async def extract_language_code(self, url: str, language: str) -> CodeExtractionResult:
        """
        Extract code for a specific language from a page with interactive widgets.
        
        Args:
            url: URL of the page to crawl
            language: Programming language to extract (e.g., 'python', 'javascript')
            
        Returns:
            CodeExtractionResult with the extracted code
        """
        
        crawler = AsyncWebCrawler(config=self.browser_config)
        
        async with crawler:
            try:
                # Generate language-specific JavaScript
                extraction_js = self._generate_language_specific_js(language)
                
                # Configure crawler with our JavaScript
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    stream=False,
                    wait_for=None,  # Don't wait for networkidle
                    delay_before_return_html=3.0,
                    js_code=[extraction_js],
                    wait_for_images=False,
                    screenshot=False,
                    verbose=True,
                    page_timeout=30000
                )
                
                # Crawl the page
                result = await crawler.arun(url=url, config=config)
                
                if result.success:
                    # The JavaScript stores result in window.codeExtractionResult
                    # We need to extract this somehow - for now, let's use code block extraction
                    # from the resulting markdown as a fallback
                    
                    code_blocks = extract_code_blocks(result.markdown, min_length=50)
                    
                    # Look for the longest code block (likely our target)
                    best_code = ""
                    for block in code_blocks:
                        if len(block['code']) > len(best_code):
                            # Check if this looks like the target language
                            code_content = block['code'].lower()
                            language_indicators = {
                                'python': ['import ', 'def ', 'print(', 'from ', '#!/usr/bin/python'],
                                'javascript': ['function', 'const ', 'let ', 'var ', 'console.log', '=>'],
                                'shell': ['#!/bin/bash', 'curl ', 'wget ', '$ ', 'export '],
                                'go': ['package ', 'import ', 'func ', 'go ', 'fmt.'],
                                'kotlin': ['package ', 'import ', 'fun ', 'suspend ', 'kotlinx'],
                                'java': ['public class', 'import java', 'public static', 'System.'],
                                'php': ['<?php', '$', 'function ', 'echo ', 'require'],
                                'ruby': ['def ', 'require ', 'puts ', 'class ', 'end']
                            }
                            
                            # Check if code matches language patterns
                            if language.lower() in language_indicators:
                                indicators = language_indicators[language.lower()]
                                if any(indicator in code_content for indicator in indicators):
                                    best_code = block['code']
                                    break
                            else:
                                # No specific patterns, just take the longest
                                best_code = block['code']
                    
                    if best_code:
                        return CodeExtractionResult(
                            language=language,
                            code=best_code,
                            length=len(best_code),
                            success=True
                        )
                    else:
                        return CodeExtractionResult(
                            language=language,
                            code="",
                            length=0,
                            success=False,
                            error="No code content found for this language"
                        )
                else:
                    return CodeExtractionResult(
                        language=language,
                        code="",
                        length=0,
                        success=False,
                        error=f"Crawl failed: {result.error_message}"
                    )
                    
            except Exception as e:
                return CodeExtractionResult(
                    language=language,
                    code="",
                    length=0,
                    success=False,
                    error=str(e)
                )

    async def extract_all_languages(self, url: str, languages: Optional[List[str]] = None) -> WidgetExtractionSummary:
        """
        Extract code examples for multiple languages from a page.
        
        Args:
            url: URL of the page to crawl
            languages: List of languages to extract (defaults to common languages)
            
        Returns:
            WidgetExtractionSummary with all extraction results
        """
        
        if languages is None:
            languages = ['shell', 'python', 'javascript', 'go', 'kotlin']
        
        print(f"Extracting code for languages: {languages}")
        
        # Extract each language sequentially to avoid conflicts
        results = []
        for language in languages:
            print(f"Extracting {language}...")
            result = await self.extract_language_code(url, language)
            results.append(result)
            
            if result.success:
                print(f"✅ {language}: {result.length} characters")
            else:
                print(f"❌ {language}: {result.error}")
            
            # Small delay between extractions
            await asyncio.sleep(1)
        
        # Calculate summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_length = sum(r.length for r in results if r.success)
        
        return WidgetExtractionSummary(
            url=url,
            total_languages_attempted=len(languages),
            successful_extractions=successful,
            failed_extractions=failed,
            results=results,
            total_code_length=total_length
        )

    def store_extracted_codes(self, summary: WidgetExtractionSummary) -> Dict[str, Any]:
        """
        Store the extracted code examples in Supabase.
        
        Args:
            summary: WidgetExtractionSummary from extract_all_languages
            
        Returns:
            Dictionary with storage results
        """
        
        try:
            supabase_client = get_supabase_client()
            
            # Prepare data for successful extractions only
            successful_results = [r for r in summary.results if r.success and r.code.strip()]
            
            if not successful_results:
                return {
                    "success": False,
                    "error": "No successful code extractions to store"
                }
            
            # Extract source_id from URL
            parsed_url = urlparse(summary.url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Prepare lists for batch insertion
            urls = []
            chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            # Process each successful extraction
            for i, result in enumerate(successful_results):
                urls.append(summary.url)
                chunk_numbers.append(i)
                code_examples.append(result.code)
                
                # Generate summary for this code example
                summary_text = generate_code_example_summary(
                    result.code, 
                    f"Code example for {result.language}", 
                    f"Programming language: {result.language}"
                )
                code_summaries.append(summary_text)
                
                # Create metadata
                metadata = {
                    "chunk_index": i,
                    "url": summary.url,
                    "source": source_id,
                    "char_count": result.length,
                    "word_count": len(result.code.split()),
                    "language": result.language,
                    "extraction_method": "dedicated_widget_crawler",
                    "widget_extracted": True
                }
                code_metadatas.append(metadata)
            
            # Store in Supabase
            add_code_examples_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                code_examples,
                code_summaries,
                code_metadatas
            )
            
            return {
                "success": True,
                "stored_examples": len(successful_results),
                "languages": [r.language for r in successful_results],
                "total_characters": sum(r.length for r in successful_results),
                "source_id": source_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Convenience functions for easy usage
async def extract_single_language(url: str, language: str) -> CodeExtractionResult:
    """Extract code for a single language from a URL."""
    crawler = CodeWidgetCrawler()
    return await crawler.extract_language_code(url, language)


async def extract_multiple_languages(url: str, languages: List[str]) -> WidgetExtractionSummary:
    """Extract code for multiple languages from a URL."""
    crawler = CodeWidgetCrawler()
    return await crawler.extract_all_languages(url, languages)


async def extract_and_store(url: str, languages: List[str]) -> Dict[str, Any]:
    """Extract code for multiple languages and store in Supabase."""
    crawler = CodeWidgetCrawler()
    summary = await crawler.extract_all_languages(url, languages)
    storage_result = crawler.store_extracted_codes(summary)
    
    return {
        "extraction_summary": {
            "url": summary.url,
            "attempted": summary.total_languages_attempted,
            "successful": summary.successful_extractions,
            "failed": summary.failed_extractions,
            "total_code_length": summary.total_code_length
        },
        "storage_result": storage_result,
        "detailed_results": [
            {
                "language": r.language,
                "success": r.success,
                "length": r.length,
                "error": r.error
            }
            for r in summary.results
        ]
    }