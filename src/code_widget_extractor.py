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
    
    def generate_extraction_javascript(self, widgets: List[CodeWidget]) -> str:
        """
        Generate JavaScript code to interact with detected widgets and extract all content.
        """
        
        js_code_parts = [
            "// Generic Code Widget Extraction",
            "console.log('Starting code widget extraction...');",
            "",
            "// Wait for page to be fully loaded",
            "await new Promise(r => setTimeout(r, 1000));",
            ""
        ]
        
        for i, widget in enumerate(widgets):
            js_code_parts.extend([
                f"// Extract widget {i+1}: {widget.title}",
                f"console.log('Processing widget: {widget.title}');",
                ""
            ])
            
            if widget.widget_type == "tabs":
                js_code_parts.extend(self._generate_tab_extraction_js(widget))
            elif widget.widget_type == "dropdown":
                js_code_parts.extend(self._generate_dropdown_extraction_js(widget))
            else:
                js_code_parts.extend(self._generate_generic_extraction_js(widget))
            
            js_code_parts.append("")
        
        # Add final wait and completion marker
        js_code_parts.extend([
            "// Final wait for all content to load",
            "await new Promise(r => setTimeout(r, 1000));",
            "console.log('Code widget extraction completed');",
            "",
            "// Mark extraction as complete",
            "window.codeWidgetExtractionComplete = true;"
        ])
        
        return "\n".join(js_code_parts)
    
    def _generate_tab_extraction_js(self, widget: CodeWidget) -> List[str]:
        """Generate JavaScript for tab-based widgets."""
        
        js_lines = []
        
        # Try multiple selector strategies
        tab_selectors = list(set([
            *widget.selectors.values(),
            *self.common_patterns["tab_selectors"]
        ]))
        
        for selector in tab_selectors[:3]:  # Try top 3 selectors
            js_lines.extend([
                f"// Try selector: {selector}",
                f"try {{",
                f"    const tabs = document.querySelectorAll('{selector}');",
                f"    if (tabs.length > 0) {{",
                f"        console.log(`Found ${{tabs.length}} tabs with selector {selector}`);",
                f"        for (let tab of tabs) {{",
                f"            // Check if this looks like a language tab",
                f"            const tabText = tab.textContent || tab.innerText || '';",
                f"            const hasLangName = /{'/'.join(self.common_patterns['language_indicators'])}/i.test(tabText);",
                f"            if (hasLangName) {{",
                f"                console.log(`Clicking tab: ${{tabText}}`);",
                f"                tab.click();",
                f"                await new Promise(r => setTimeout(r, 500));",
                f"            }}",
                f"        }}",
                f"    }}",
                f"}} catch (e) {{",
                f"    console.log(`Error with selector {selector}:`, e);",
                f"}}",
                ""
            ])
        
        return js_lines
    
    def _generate_dropdown_extraction_js(self, widget: CodeWidget) -> List[str]:
        """Generate JavaScript for dropdown-based widgets."""
        
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
    
    def _generate_generic_extraction_js(self, widget: CodeWidget) -> List[str]:
        """Generate JavaScript for unknown widget types."""
        
        return [
            "// Generic widget interaction",
            "const clickableElements = document.querySelectorAll('button, [role=\"button\"], .btn, [data-toggle]');",
            "for (let element of clickableElements) {",
            "    const text = element.textContent || element.innerText || '';",
            f"    const hasLangName = /{'/'.join(self.common_patterns['language_indicators'])}/i.test(text);",
            "    if (hasLangName) {",
            "        console.log(`Clicking element: ${text}`);",
            "        element.click();",
            "        await new Promise(r => setTimeout(r, 300));",
            "    }",
            "}"
        ]


def create_enhanced_crawler_config(extractor: CodeWidgetExtractor, 
                                 widgets: List[CodeWidget]) -> CrawlerRunConfig:
    """
    Create a Crawl4AI configuration with widget extraction JavaScript.
    """
    
    extraction_js = extractor.generate_extraction_javascript(widgets)
    
    return CrawlerRunConfig(
        cache_mode="bypass",
        stream=False,
        wait_for="networkidle",
        delay_before_return_html=3.0,  # Wait for interactions to complete
        js_code=[extraction_js],
        wait_for_images=False,
        screenshot=False,
        verbose=True
    )