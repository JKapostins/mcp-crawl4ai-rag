"""
Interactive Code Widget Extractor

A generic solution for extracting code examples from interactive widgets
found on documentation websites. Uses LLM reasoning to detect widget patterns
and automatically extract all language variants.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from crawl4ai import CrawlerRunConfig


@dataclass
class CodeWidget:
    """Represents a detected code widget with multiple language variants."""
    title: str
    description: str
    languages: List[str]
    code_examples: Dict[str, str]  # language -> code
    selectors: Dict[str, str]      # language -> CSS selector
    widget_type: str               # 'tabs', 'dropdown', 'buttons', etc.


@dataclass
class WidgetDetectionResult:
    """Result of widget detection analysis."""
    widgets_found: List[CodeWidget]
    javascript_code: str
    confidence_score: float
    extraction_strategy: str


class CodeWidgetExtractor:
    """
    Generic extractor for interactive code widgets using LLM reasoning.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the code widget extractor.
        
        Args:
            model_name: OpenAI model to use for analysis (defaults to MODEL_CHOICE env var)
        """
        self.model_name = model_name or os.getenv("MODEL_CHOICE", "gpt-4o-mini")
        
        # Common patterns found across documentation sites
        self.common_patterns = {
            "tab_selectors": [
                "[role='tab']",
                ".tab",
                ".tab-button", 
                ".code-tab",
                "[data-tab]",
                "[data-language]",
                ".language-tab",
                "[aria-selected]",
                ".nav-tabs a",
                ".tabbed-code-block button"
            ],
            "content_selectors": [
                "[role='tabpanel']",
                ".tab-content",
                ".tab-pane",
                ".code-content",
                ".highlight",
                "pre code",
                ".code-example"
            ],
            "language_indicators": [
                "shell", "bash", "python", "javascript", "js", "go", "kotlin",
                "java", "php", "ruby", "rust", "typescript", "ts", "curl",
                "http", "json", "yaml", "xml", "sql", "powershell", "cmd"
            ]
        }
    
    async def detect_code_widgets(self, html_content: str, url: str = "") -> WidgetDetectionResult:
        """
        Use LLM to analyze HTML and detect interactive code widgets.
        
        Args:
            html_content: Raw HTML content of the page
            url: URL for context (optional)
            
        Returns:
            WidgetDetectionResult with detected widgets and extraction strategy
        """
        
        # Create analysis prompt
        prompt = self._create_detection_prompt(html_content, url)
        
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert web scraper specialized in detecting interactive code widgets on documentation websites."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_detection_result(result_text)
            
        except Exception as e:
            print(f"Error in LLM widget detection: {e}")
            return self._fallback_detection(html_content)
    
    def _create_detection_prompt(self, html_content: str, url: str) -> str:
        """Create the LLM prompt for widget detection."""
        
        # Truncate HTML to avoid token limits
        html_snippet = html_content[:8000] if len(html_content) > 8000 else html_content
        
        return f"""
Analyze this HTML content from {url} to detect interactive code widgets that show examples in multiple programming languages.

HTML Content:
```html
{html_snippet}
```

Look for:
1. **Tab-based widgets** with language names (Shell, Python, Go, JavaScript, etc.)
2. **Dropdown selectors** for choosing programming languages  
3. **Button groups** for switching between code examples
4. **Any interactive elements** that reveal different code examples

For each widget found, identify:
- Widget type (tabs/dropdown/buttons/other)
- Programming languages available
- CSS selectors for language tabs/buttons
- CSS selectors for code content areas
- Any special interaction requirements

Common patterns to look for:
- Elements with role="tab", data-language, data-tab attributes
- Class names containing "tab", "language", "code"
- Text content mentioning programming languages
- Multiple code blocks with similar structure

Respond in this JSON format:
{{
    "widgets_found": [
        {{
            "title": "Widget description",
            "widget_type": "tabs|dropdown|buttons|other",
            "languages": ["python", "javascript", "shell", "go"],
            "tab_selectors": {{
                "python": "CSS selector for Python tab",
                "javascript": "CSS selector for JS tab"
            }},
            "content_selectors": {{
                "python": "CSS selector for Python code content",
                "javascript": "CSS selector for JS code content"
            }},
            "interaction_method": "click|hover|other"
        }}
    ],
    "javascript_strategy": "JavaScript code to execute for extraction",
    "confidence": 0.0-1.0,
    "notes": "Any special considerations or fallback approaches"
}}

If no widgets are detected, return confidence 0.0 and empty widgets_found array.
"""

    def _parse_detection_result(self, result_text: str) -> WidgetDetectionResult:
        """Parse LLM response into structured result."""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group())
            
            widgets = []
            for widget_data in data.get("widgets_found", []):
                widget = CodeWidget(
                    title=widget_data.get("title", "Code Widget"),
                    description=widget_data.get("title", ""),
                    languages=widget_data.get("languages", []),
                    code_examples={},  # Will be populated during extraction
                    selectors=widget_data.get("tab_selectors", {}),
                    widget_type=widget_data.get("widget_type", "unknown")
                )
                widgets.append(widget)
            
            return WidgetDetectionResult(
                widgets_found=widgets,
                javascript_code=data.get("javascript_strategy", ""),
                confidence_score=data.get("confidence", 0.0),
                extraction_strategy=data.get("notes", "")
            )
            
        except Exception as e:
            print(f"Error parsing LLM detection result: {e}")
            return self._fallback_detection("")
    
    def _fallback_detection(self, html_content: str) -> WidgetDetectionResult:
        """Fallback heuristic-based detection when LLM fails."""
        
        widgets = []
        javascript_parts = []
        
        # Look for common tab patterns
        for tab_selector in self.common_patterns["tab_selectors"]:
            if tab_selector.replace("[", "\\[").replace("]", "\\]") in html_content:
                # Found potential tabs - create generic widget
                languages = []
                for lang in self.common_patterns["language_indicators"]:
                    if lang.lower() in html_content.lower():
                        languages.append(lang)
                
                if languages:
                    widget = CodeWidget(
                        title="Detected Code Widget",
                        description="Heuristically detected widget",
                        languages=languages[:5],  # Limit to 5 languages
                        code_examples={},
                        selectors={lang: tab_selector for lang in languages},
                        widget_type="tabs"
                    )
                    widgets.append(widget)
                    
                    # Add generic JavaScript
                    javascript_parts.append(f"""
                    // Click all tabs matching {tab_selector}
                    const tabs = document.querySelectorAll('{tab_selector}');
                    for (let tab of tabs) {{
                        tab.click();
                        await new Promise(r => setTimeout(r, 300));
                    }}
                    """)
                    break
        
        return WidgetDetectionResult(
            widgets_found=widgets,
            javascript_code="\n".join(javascript_parts),
            confidence_score=0.3 if widgets else 0.0,
            extraction_strategy="Fallback heuristic detection"
        )
    
    def generate_extraction_javascript(self, widgets: List[CodeWidget], target_languages: List[str] = None) -> str:
        """
        Generate JavaScript code to interact with detected widgets and extract all content.
        
        Args:
            widgets: List of detected code widgets
            target_languages: List of languages to prioritize for extraction (e.g., ['python', 'javascript'])
        """
        
        js_code_parts = [
            "// Generic Code Widget Extraction",
            "console.log('Starting code widget extraction...');",
            "",
            "// Wait for page to be fully loaded",
            "await new Promise(r => setTimeout(r, 1000));",
            ""
        ]
        
        # Set up target languages (use provided or default)
        if target_languages is None:
            target_languages = ['python', 'javascript', 'shell', 'go', 'kotlin']
        
        # Convert to lowercase for matching
        priority_langs = [lang.lower().strip() for lang in target_languages]
        
        js_code_parts.extend([
            f"// Target languages (in priority order): {', '.join(priority_langs)}",
            f"const targetLanguages = {priority_langs};",
            "console.log('Prioritizing languages:', targetLanguages);",
            ""
        ])
        
        for i, widget in enumerate(widgets):
            js_code_parts.extend([
                f"// Extract widget {i+1}: {widget.title}",
                f"console.log('Processing widget: {widget.title}');",
                ""
            ])
            
            if widget.widget_type == "tabs":
                js_code_parts.extend(self._generate_tab_extraction_js(widget, priority_langs))
            elif widget.widget_type == "dropdown":
                js_code_parts.extend(self._generate_dropdown_extraction_js(widget, priority_langs))
            else:
                js_code_parts.extend(self._generate_generic_extraction_js(widget, priority_langs))
            
            js_code_parts.append("")
        
        # Add final comprehensive extraction
        lang_pattern = '|'.join(self.common_patterns['language_indicators'])
        js_code_parts.extend([
            "// Final comprehensive extraction - click all tabs quickly to load all content",
            "console.log('Starting final comprehensive extraction...');",
            "",
            "const allLanguageTabs = document.querySelectorAll(",
            "    'button[data-language], button[aria-controls*=\"code\"], button[role=\"tab\"], .tab'",
            ");",
            "",
            "for (let tab of allLanguageTabs) {",
            "    const text = (tab.textContent || tab.innerText || '').trim().toLowerCase();",
            "    const dataLang = tab.getAttribute('data-language') || '';",
            "    ",
            f"    if (/({lang_pattern})/i.test(text) || /({lang_pattern})/i.test(dataLang)) {{",
            "        console.log(`Final click: ${text} (${dataLang})`);",
            "        try {",
            "            tab.click();",
            "            await new Promise(r => setTimeout(r, 300)); // Quick clicks",
            "        } catch (e) {",
            "            console.log(`Error in final click: ${e}`);",
            "        }",
            "    }",
            "}",
            "",
            "// Wait for all content to settle",
            "await new Promise(r => setTimeout(r, 2000));",
            "",
            "console.log('Widget extraction completed');",
            "",
            "// Mark extraction as complete",
            "window.codeWidgetExtractionComplete = true;"
        ])
        
        return "\n".join(js_code_parts)
    
    def _generate_tab_extraction_js(self, widget: CodeWidget, priority_langs: List[str] = None) -> List[str]:
        """Generate JavaScript for tab-based widgets with language prioritization."""
        
        # Create language pattern for regex
        lang_pattern = '|'.join(self.common_patterns['language_indicators'])
        
        # Build prioritized language clicking logic with aggressive search
        priority_js = []
        if priority_langs:
            priority_js.extend([
                "// AGGRESSIVE language tab clicking strategy",
                "for (let targetLang of targetLanguages) {",
                "    console.log(`🎯 Looking for ${targetLang} tab/button...`);",
                "    let foundTarget = false;",
                "    ",
                "    // Strategy 1: Look for ANY element containing the language name",
                "    const allElements = document.querySelectorAll('*');",
                "    for (let element of allElements) {",
                "        const text = (element.textContent || element.innerText || '').trim().toLowerCase();",
                "        const tagName = element.tagName.toLowerCase();",
                "        ",
                "        // Check if element contains target language and is likely clickable",
                "        if (text === targetLang || text === targetLang.toUpperCase() || ",
                "            (text.length <= 20 && text.includes(targetLang))) {",
                "            ",
                "            // Check if it's a clickable element",
                "            const isClickable = tagName === 'button' || tagName === 'a' ||",
                "                               element.onclick || element.getAttribute('onclick') ||",
                "                               element.style.cursor === 'pointer' ||",
                "                               element.getAttribute('role') === 'button' ||",
                "                               element.classList.contains('btn') ||",
                "                               element.classList.contains('tab') ||",
                "                               element.classList.contains('clickable');",
                "            ",
                "            if (isClickable) {",
                "                console.log(`📍 Found ${targetLang} clickable: ${tagName} with text '${text.substring(0,30)}'`);",
                "                try {",
                "                    element.click();",
                "                    await new Promise(r => setTimeout(r, 2000)); // Wait longer for content",
                "                    ",
                "                    // Capture all code content after clicking",
                "                    const allCodeBlocks = document.querySelectorAll('pre, code, .highlight, [class*=\"code\"], .code-example');",
                "                    let capturedCode = '';",
                "                    ",
                "                    allCodeBlocks.forEach(block => {",
                "                        const code = block.textContent || block.innerText || '';",
                "                        if (code.length > 30 && !capturedCode.includes(code.substring(0, 100))) {",
                "                            capturedCode += code + '\\n\\n';",
                "                        }",
                "                    });",
                "                    ",
                "                    if (capturedCode.length > 50) {",
                "                        // Inject captured code into page",
                "                        const codeContainer = document.createElement('div');",
                "                        codeContainer.className = 'extracted-widget-code';",
                "                        codeContainer.innerHTML = `",
                "                            <h2>EXTRACTED: ${targetLang.toUpperCase()}</h2>",
                "                            <pre><code class='language-${targetLang}'>${capturedCode}</code></pre>",
                "                            <p>Language: ${targetLang} | Extracted: ${capturedCode.length} chars</p>",
                "                            <hr/>",
                "                        `;",
                "                        document.body.appendChild(codeContainer);",
                "                        console.log(`✅ EXTRACTED ${capturedCode.length} chars for ${targetLang}`);",
                "                        foundTarget = true;",
                "                    }",
                "                    ",
                "                } catch (e) {",
                "                    console.log(`❌ Error clicking ${targetLang} element: ${e}`);",
                "                }",
                "            }",
                "        }",
                "    }",
                "    ",
                "    // Strategy 2: Try clicking any element with language name in attributes",
                "    if (!foundTarget) {",
                "        const elementsWithLangAttrs = document.querySelectorAll(`[*|*=\"${targetLang}\"], [*|*=\"${targetLang.toUpperCase()}\"]`);",
                "        for (let element of elementsWithLangAttrs) {",
                "            console.log(`🔍 Trying attribute-based element for ${targetLang}`);",
                "            try {",
                "                element.click();",
                "                await new Promise(r => setTimeout(r, 1500));",
                "                foundTarget = true;",
                "                break;",
                "            } catch (e) {",
                "                console.log(`Failed: ${e}`);",
                "            }",
                "        }",
                "    }",
                "    ",
                "    if (!foundTarget) {",
                "        console.log(`❌ No ${targetLang} element found`);",
                "    }",
                "}",
                "",
                "// Final wait for all interactions to complete",
                "await new Promise(r => setTimeout(r, 2000));",
                ""
            ])
        
        return priority_js + [
            f"// Extract {widget.widget_type} widget: {widget.title}",
            f"console.log('Processing {widget.widget_type} widget...');",
            "",
            "// Strategy 1: Look for specific language buttons",
            "const languageButtons = document.querySelectorAll(",
            "    'button[data-language], button[aria-controls*=\"code\"], button[role=\"tab\"], .tab, .language-tab, .code-tab'",
            ");",
            "",
            "console.log(`Found ${languageButtons.length} potential language buttons`);",
            "",
            "for (let button of languageButtons) {",
            "    const text = (button.textContent || button.innerText || '').trim().toLowerCase();",
            "    const dataLang = button.getAttribute('data-language') || '';",
            "    const ariaControls = button.getAttribute('aria-controls') || '';",
            "    ",
            f"    // Check if this looks like a programming language",
            f"    const isLanguageTab = /({lang_pattern})/i.test(text) || /({lang_pattern})/i.test(dataLang);",
            "    ",
            "    if (isLanguageTab) {",
            "        console.log(`Clicking language button: ${text} (${dataLang})`);",
            "        try {",
            "            button.click();",
            "            await new Promise(r => setTimeout(r, 1200)); // Wait for content to load",
            "            ",
            "            // Capture code content after this tab click",
            "            const codeBlocks = document.querySelectorAll('pre code, .highlight code, .code-content code, [class*=\"code\"] code, .code-example code');",
            "            let capturedAny = false;",
            "            ",
            "            codeBlocks.forEach((block, idx) => {",
            "                const code = block.textContent || block.innerText || '';",
            "                if (code.length > 50) {",
            "                    // Create a visible code block in the page",
            "                    const newCodeBlock = document.createElement('div');",
            "                    newCodeBlock.className = 'widget-extracted-code';",
            "                    newCodeBlock.innerHTML = `",
            "                        <h3>Widget Code Example: ${text || dataLang || 'Unknown'}</h3>",
            "                        <pre><code class='language-${(text || dataLang || '').toLowerCase()}'>${code}</code></pre>",
            "                        <p>Language: ${text || dataLang} | Length: ${code.length} chars</p>",
            "                    `;",
            "                    ",
            "                    // Append to the end of the page so it gets picked up by markdown",
            "                    document.body.appendChild(newCodeBlock);",
            "                    capturedAny = true;",
            "                    console.log(`Injected ${code.length} chars of ${text} code into page`);",
            "                }",
            "            });",
            "            ",
            "            if (!capturedAny) {",
            "                console.log(`No code content found after clicking ${text} tab`);",
            "            }",
            "        } catch (e) {",
            "            console.log(`Error clicking button: ${e}`);",
            "        }",
            "    }",
            "}",
            "",
            "// Strategy 2: Click any element containing language names",
            "const allClickable = document.querySelectorAll('*[onclick], button, [role=\"button\"], .clickable');",
            "for (let el of allClickable) {",
            "    const text = (el.textContent || el.innerText || '').trim();",
            f"    if (/^(shell|python|javascript|go|kotlin|java|php|ruby|curl)$/i.test(text)) {{",
            "        console.log(`Found language element: ${text}`);",
            "        try {",
            "            el.click();",
            "            await new Promise(r => setTimeout(r, 500));",
            "        } catch (e) {",
            "            console.log(`Error clicking element: ${e}`);",
            "        }",
            "    }",
            "}",
            ""
        ]
    
    def _generate_dropdown_extraction_js(self, widget: CodeWidget, priority_langs: List[str] = None) -> List[str]:
        """Generate JavaScript for dropdown-based widgets with language prioritization."""
        
        return [
            "// Handle dropdown widget",
            "const dropdowns = document.querySelectorAll('select, .dropdown, [role=\"combobox\"]');",
            "for (let dropdown of dropdowns) {",
            "    // Try to expand dropdown and click options",
            "    dropdown.click();",
            "    await new Promise(r => setTimeout(r, 300));",
            "    ",
            "    const options = dropdown.querySelectorAll('option, .dropdown-item, [role=\"option\"]');",
            "    for (let option of options) {",
            "        option.click();",
            "        await new Promise(r => setTimeout(r, 300));",
            "    }",
            "}"
        ]
    
    def _generate_generic_extraction_js(self, widget: CodeWidget, priority_langs: List[str] = None) -> List[str]:
        """Generate JavaScript for unknown widget types."""
        
        lang_pattern = '|'.join(self.common_patterns['language_indicators'])
        
        return [
            "// Generic widget interaction",
            "console.log('Trying generic widget interaction...');",
            "const clickableElements = document.querySelectorAll('button, [role=\"button\"], .btn, [data-toggle], .clickable');",
            "for (let element of clickableElements) {",
            "    const text = (element.textContent || element.innerText || '').trim();",
            f"    const hasLangName = /({lang_pattern})/i.test(text);",
            "    if (hasLangName) {",
            "        console.log(`Clicking generic element: ${text}`);",
            "        try {",
            "            element.click();",
            "            await new Promise(r => setTimeout(r, 300));",
            "        } catch (e) {",
            "            console.log(`Error clicking: ${e}`);",
            "        }",
            "    }",
            "}"
        ]


def create_enhanced_crawler_config(extractor: CodeWidgetExtractor, 
                                 widgets: List[CodeWidget],
                                 target_languages: List[str] = None) -> CrawlerRunConfig:
    """
    Create a Crawl4AI configuration with widget extraction JavaScript.
    
    Args:
        extractor: The CodeWidgetExtractor instance
        widgets: List of detected widgets
        target_languages: List of languages to prioritize for extraction
    """
    
    extraction_js = extractor.generate_extraction_javascript(widgets, target_languages)
    
    return CrawlerRunConfig(
        cache_mode="bypass",
        stream=False,
        wait_for=None,  # Don't wait for networkidle to avoid timeout
        delay_before_return_html=5.0,  # Wait longer for interactions to complete
        js_code=[extraction_js],
        wait_for_images=False,
        screenshot=False,
        verbose=True,
        page_timeout=30000  # 30 second timeout instead of default 60s
    )