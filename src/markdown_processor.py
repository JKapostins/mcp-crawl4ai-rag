#!/usr/bin/env python3

"""
Standalone markdown processor for local files
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from utils import extract_code_blocks_with_llm, smart_chunk_markdown, chunk_document_llm, chunk_document_single


async def clean_markdown_for_rag_simple(markdown_content: str) -> str:
    """
    Simple markdown cleaning that doesn't require external dependencies.
    This is a fallback when the full LLM-based cleaning isn't available.
    """
    # Basic cleaning - remove excessive whitespace, normalize line endings
    lines = markdown_content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Remove excessive whitespace
        cleaned_line = line.strip()
        # Skip empty lines that are repeated
        if cleaned_line or (cleaned_lines and cleaned_lines[-1]):
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


async def process_local_markdown_standalone(
    file_path: str,
    chunking_strategy: str = "character",
    chunk_size: int = 5000,
    chunking_prompt: Optional[str] = None,
    title: Optional[str] = None,
    source_name: Optional[str] = None,
    supabase_client=None
) -> str:
    """
    Standalone version of process_local_markdown that doesn't depend on MCP framework.
    """
    try:
        # Validate file path
        if not os.path.exists(file_path):
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": f"File not found: {file_path}"
            }, indent=2)
        
        # Read the markdown file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": f"Error reading file: {str(e)}"
            }, indent=2)
        
        if not content.strip():
            return json.dumps({
                "success": False,
                "file_path": file_path,
                "error": "File is empty"
            }, indent=2)
        
        # Set defaults
        if not title:
            title = Path(file_path).stem
        if not source_name:
            source_name = file_path
        
        # Use provided source_name or create one from filename (needed for summary generation)
        final_source_id = source_name if source_name else f"local_markdown_{Path(file_path).stem}"
        
        # Clean content for RAG if enabled (use full agentic RAG like web tools)
        use_agentic_rag = os.getenv("USE_AGENTIC_RAG", "false").lower() == "true"
        use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false").lower() == "true"
        cleaned_content = content
        
        if use_agentic_rag:
            try:
                # Use OpenAI directly for agentic RAG (avoiding MCP dependency)
                import openai
                
                openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                rag_prompt = f"""
Clean and optimize this markdown content for RAG (Retrieval Augmented Generation) consumption.

Your task is to:
1. Structure the content for easy chunking and semantic search
2. Preserve all technical details and code examples
3. Add context to code blocks and examples
4. Consolidate fragmented information
5. Create clear hierarchical structure
6. Remove navigation elements, ads, or non-content

Source: {f"file://{file_path}"}

Content to optimize:
{content}

Return only the cleaned and optimized markdown content."""

                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": rag_prompt}],
                    temperature=0.1
                )
                
                cleaned_content = response.choices[0].message.content
                print("✅ Applied full agentic RAG content optimization")
                
            except Exception as e:
                print(f"Warning: Failed to apply full agentic RAG, using simple cleaning: {e}")
                try:
                    cleaned_content = await clean_markdown_for_rag_simple(content)
                except Exception as e2:
                    print(f"Warning: Simple cleaning also failed: {e2}")
                    cleaned_content = content
        
        # Extract code blocks with LLM
        code_blocks = []
        try:
            code_blocks = extract_code_blocks_with_llm(cleaned_content, f"file://{file_path}")
        except Exception as e:
            print(f"Warning: Failed to extract code blocks with LLM: {e}")
            # Fallback to simple regex-based extraction
            import re
            code_block_pattern = r'```(\w+)?\s*\n(.*?)\n```'
            matches = re.findall(code_block_pattern, cleaned_content, re.DOTALL)
            code_blocks = [
                {
                    "code": match[1].strip(),
                    "language": match[0] if match[0] else "unknown",
                    "summary": f"Code block in {match[0] if match[0] else 'unknown'} language",
                    "context": "Extracted from markdown file",
                    "type": "code_block"
                }
                for match in matches
            ]
        
        # Chunk the content based on strategy
        chunks = []
        if chunking_strategy == "document":
            chunk_list = chunk_document_single(cleaned_content)
            
            # For document strategy, add document summary at the top like scrape_single_page does
            if chunk_list and len(chunk_list) > 0:
                try:
                    # Generate document summary to add at the top
                    from utils import extract_source_summary
                    document_summary = extract_source_summary(final_source_id, cleaned_content, max_length=300)
                    
                    # Add summary at the top of the document chunk
                    enhanced_content = f"# Document Summary\n\n{document_summary}\n\n---\n\n{chunk_list[0]}"
                    chunks = [{"content": enhanced_content, "chunk_index": 0}]
                    print(f"✅ Added document summary to chunk (document strategy)")
                    
                except Exception as e:
                    print(f"Warning: Could not generate chunk summary: {e}")
                    chunks = [{"content": chunk, "chunk_index": i} for i, chunk in enumerate(chunk_list)]
            else:
                chunks = [{"content": chunk, "chunk_index": i} for i, chunk in enumerate(chunk_list)]
        elif chunking_strategy == "llm":
            try:
                chunk_list = chunk_document_llm(cleaned_content, chunking_prompt, f"file://{file_path}")
                chunks = [{"content": chunk, "chunk_index": i} for i, chunk in enumerate(chunk_list)]
            except Exception as e:
                print(f"Warning: LLM chunking failed, falling back to character chunking: {e}")
                chunk_list = smart_chunk_markdown(cleaned_content, chunk_size)
                chunks = [{"content": chunk, "chunk_index": i} for i, chunk in enumerate(chunk_list)]
        else:  # character (default)
            chunk_list = smart_chunk_markdown(cleaned_content, chunk_size)
            chunks = [{"content": chunk, "chunk_index": i} for i, chunk in enumerate(chunk_list)]
        
        # Mock storage if no client provided
        source_id = "mock-source-id"
        stored_chunks = len(chunks)
        stored_code_blocks = len(code_blocks)
        
        if supabase_client:
            # Check if source already exists
            existing_source = None
            try:
                existing_result = supabase_client.table("sources").select("*").eq("source_id", final_source_id).execute()
                if existing_result.data:
                    existing_source = existing_result.data[0]
                    source_id = existing_source["source_id"]
                    print(f"Using existing source: {source_id}")
            except Exception as e:
                print(f"Warning: Could not check for existing source: {e}")
            
            # Create new source if it doesn't exist
            if not existing_source:
                # Generate document summary
                try:
                    from utils import extract_source_summary
                    document_summary = extract_source_summary(final_source_id, cleaned_content, max_length=500)
                except Exception as e:
                    print(f"Warning: Could not generate document summary: {e}")
                    document_summary = f"Local markdown file: {title}. Processing method: local_markdown. Chunking: {chunking_strategy}"
                
                source_data = {
                    "source_id": final_source_id,
                    "summary": document_summary,
                    "total_word_count": len(cleaned_content.split()),
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                try:
                    source_result = supabase_client.table("sources").insert(source_data).execute()
                    source_id = source_result.data[0]["source_id"]
                    print(f"Created new source: {source_id}")
                except Exception as e:
                    return json.dumps({
                        "success": False,
                        "file_path": file_path,
                        "error": f"Failed to store source: {str(e)}"
                    }, indent=2)
            
            # Store chunks with advanced embedding processing like web tools
            stored_chunks = 0
            
            # Prepare chunks for advanced processing
            chunk_texts = [chunk["content"] for chunk in chunks]
            chunk_embeddings = []
            
            # Generate embeddings using the same functions as web tools
            try:
                from utils import create_embeddings_batch, generate_contextual_embedding, process_chunk_with_context
                from concurrent.futures import ThreadPoolExecutor
                import time
                
                if use_contextual_embeddings and len(chunk_texts) > 0:
                    print("🧠 Generating contextual embeddings...")
                    
                    # Use the same parallel processing as web tools
                    chunk_args = [(f"file://{file_path}", chunk_text, cleaned_content) for chunk_text in chunk_texts]
                    
                    with ThreadPoolExecutor(max_workers=min(4, len(chunk_texts))) as executor:
                        contextual_results = list(executor.map(process_chunk_with_context, chunk_args))
                    
                    # Extract contextual content and generate embeddings
                    contextual_texts = [result[0] if result[1] else text for result, text in zip(contextual_results, chunk_texts)]
                    chunk_embeddings = create_embeddings_batch(contextual_texts)
                    print(f"✅ Generated {len(chunk_embeddings)} contextual embeddings")
                    
                else:
                    # Standard embeddings without context
                    chunk_embeddings = create_embeddings_batch(chunk_texts)
                    print(f"✅ Generated {len(chunk_embeddings)} standard embeddings")
                    
            except Exception as e:
                print(f"Warning: Failed to generate embeddings: {e}")
                # Continue without embeddings
                chunk_embeddings = [None] * len(chunks)
            
            # Store chunks with embeddings
            for i, chunk in enumerate(chunks):
                try:
                    chunk_data = {
                        "source_id": source_id,
                        "url": f"file://{file_path}",
                        "chunk_number": chunk["chunk_index"],
                        "content": chunk["content"],
                        "metadata": {
                            "file_path": file_path,
                            "title": title,
                            "chunking_strategy": chunking_strategy,
                            "chunk_size": chunk_size if chunking_strategy == "character" else None,
                            "processing_method": "local_markdown",
                            "agentic_rag_used": use_agentic_rag,
                            "contextual_embeddings_used": use_contextual_embeddings,
                            "embedding_model": "text-embedding-3-small"
                        },
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    # Add embedding if available
                    if i < len(chunk_embeddings) and chunk_embeddings[i] is not None:
                        chunk_data["embedding"] = chunk_embeddings[i]
                    
                    supabase_client.table("crawled_pages").insert(chunk_data).execute()
                    stored_chunks += 1
                except Exception as e:
                    print(f"Warning: Failed to store chunk {chunk['chunk_index']}: {e}")
            
            # Store code blocks with embeddings - using actual schema
            stored_code_blocks = 0
            
            # Generate embeddings for code blocks
            code_embeddings = []
            if code_blocks:
                try:
                    # Create combined text for embedding (code + summary + context)
                    code_texts = []
                    for code_block in code_blocks:
                        combined_text = f"Language: {code_block.get('language', 'unknown')}\n"
                        combined_text += f"Summary: {code_block.get('summary', '')}\n"
                        combined_text += f"Context: {code_block.get('context', '')}\n"
                        combined_text += f"Code:\n{code_block['code']}"
                        code_texts.append(combined_text)
                    
                    code_embeddings = create_embeddings_batch(code_texts)
                    print(f"✅ Generated {len(code_embeddings)} code block embeddings")
                    
                except Exception as e:
                    print(f"Warning: Failed to generate code embeddings: {e}")
                    code_embeddings = [None] * len(code_blocks)
            
            for i, code_block in enumerate(code_blocks):
                try:
                    code_data = {
                        "source_id": source_id,
                        "url": f"file://{file_path}",
                        "chunk_number": i,
                        "content": code_block["code"],
                        "summary": code_block.get("summary", f"Code block in {code_block.get('language', 'unknown')} language"),
                        "metadata": {
                            "language": code_block.get("language", ""),
                            "context": code_block.get("context", ""),
                            "type": code_block.get("type", "code_block"),
                            "extraction_method": "llm",
                            "file_path": file_path,
                            "embedding_model": "text-embedding-3-small"
                        },
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    # Add embedding if available
                    if i < len(code_embeddings) and code_embeddings[i] is not None:
                        code_data["embedding"] = code_embeddings[i]
                    
                    supabase_client.table("code_examples").insert(code_data).execute()
                    stored_code_blocks += 1
                except Exception as e:
                    print(f"Warning: Failed to store code block: {e}")
            
            # Update source with final summary and word count
            try:
                if existing_source:
                    # Add to existing word count
                    new_word_count = existing_source.get("total_word_count", 0) + len(cleaned_content.split())
                    updated_summary = existing_source.get("summary", "") + f" | Added: {title} ({len(cleaned_content.split())} words, {stored_chunks} chunks, {stored_code_blocks} code blocks)"
                else:
                    # New source
                    new_word_count = len(cleaned_content.split()) 
                    updated_summary = f"Local markdown source: {final_source_id}. File: {title} ({stored_chunks} chunks, {stored_code_blocks} code blocks)"
                
                supabase_client.table("sources").update({
                    "summary": updated_summary,
                    "total_word_count": new_word_count,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("source_id", source_id).execute()
            except Exception as e:
                print(f"Warning: Failed to update source summary: {e}")
        
        return json.dumps({
            "success": True,
            "file_path": file_path,
            "title": title,
            "source_name": source_name,
            "source_id": source_id,
            "content_chunks_stored": stored_chunks,
            "code_blocks_found": stored_code_blocks,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size if chunking_strategy == "character" else None,
            "agentic_rag_used": use_agentic_rag,
            "contextual_embeddings_used": use_contextual_embeddings,
            "embeddings_generated": len([e for e in chunk_embeddings if e is not None]) if 'chunk_embeddings' in locals() else 0,
            "code_embeddings_generated": len([e for e in code_embeddings if e is not None]) if 'code_embeddings' in locals() else 0,
            "advanced_features": {
                "agentic_rag": use_agentic_rag,
                "contextual_embeddings": use_contextual_embeddings,
                "llm_code_extraction": True,
                "embedding_model": "text-embedding-3-small"
            },
            "processing_time": datetime.utcnow().isoformat()
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "file_path": file_path,
            "error": str(e)
        }, indent=2)


if __name__ == "__main__":
    # Test the standalone processor
    import asyncio
    
    async def test():
        # Create test file
        test_content = """# Test Document

This is a test document.

```python
print("Hello World")
```

More content here.
"""
        
        test_file = "/tmp/test_standalone.md"
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        try:
            result = await process_local_markdown_standalone(
                file_path=test_file,
                chunking_strategy="document",
                source_name="test-doc"
            )
            print(result)
        finally:
            os.remove(test_file)
    
    asyncio.run(test())