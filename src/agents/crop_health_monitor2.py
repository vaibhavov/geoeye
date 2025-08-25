"""
Complete Crop Health Monitoring Agent
Orchestrates data collection, analysis, and reporting using LangGraph
"""

import os
import time
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict, Literal
from dataclasses import dataclass

# LangGraph and LLM - Updated import
from langgraph.graph import StateGraph, END
try:
    from langchain_ollama import ChatOllama  # Updated import
except ImportError:
    # Fallback to old import if new package not available
    from langchain_community.chat_models import ChatOllama

# Our custom modules
from data_agent import collect_field_data, DataResponse
from toolbox import AVAILABLE_TOOLS

# =========================
# STATE MANAGEMENT
# =========================

class CropHealthState(TypedDict):
    # Input parameters
    field_id: str
    crop: str
    latitude: float
    longitude: float
    radius_m: int
    das: int  # days after sowing
    
    # Runtime data
    raw_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    issues: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    
    # Workflow control
    analysis_plan: List[str]
    confidence_scores: Dict[str, float]
    health_score: int
    status: str  # 'running', 'completed', 'error'

    # Report fields
    health_status: Optional[str]
    status_emoji: Optional[str]
    ndvi_mean: Optional[float]
    issues_found: Optional[int]
    recommendations_count: Optional[int]
    alerts_count: Optional[int]
    tools_executed: Optional[int]
    
    # Metadata
    started_at: str
    finished_at: Optional[str]
    processing_time: Optional[float]
    errors: List[str]

# =========================
# LLM SETUP
# =========================

# Brain LLM for analysis synthesis
llm_brain = ChatOllama(
    model="mistral-nemo", 
    base_url="http://localhost:11434",
    temperature=0.1,
    # Updated parameters for newer version
    num_ctx=4096,
    num_predict=512
)

# Planner LLM for orchestration
llm_planner = ChatOllama(
    model="mistral-nemo",
    base_url="http://localhost:11434", 
    temperature=0,
    num_ctx=4096
)

# =========================
# AGENT NODES
# =========================

def orchestrator_node(state: CropHealthState) -> CropHealthState:
    """Plan the analysis based on crop, stage, and available data"""
    
    print(f"🎯 Planning analysis for {state['field_id']} - {state['crop']} at {state['das']} DAS")
    
    crop = state['crop'].lower()
    das = state['das']
    
    # Base analysis tools
    base_plan = ['compute_indices', 'time_series', 'spatial_analysis']
    
    # Add weather analysis
    base_plan.append('weather_data')
    
    # Crop and stage-specific analysis
    if das >= 20:  # Mature enough for stress analysis
        base_plan.extend(['water_stress', 'nutrient_analysis'])
        
    if das >= 30:  # Disease risk relevant
        base_plan.append('disease_risk')
        
    if das >= 40:  # Yield prediction possible
        base_plan.append('yield_prediction')
    
    # Always end with intervention optimization
    base_plan.append('intervention_optimizer')
    
    state['analysis_plan'] = base_plan
    state['status'] = 'running'
    state['started_at'] = datetime.now().isoformat()
    state['errors'] = []
    
    print(f"📋 Analysis plan: {len(base_plan)} tools - {base_plan}")
    
    return state

def data_ingestion_node(state: CropHealthState) -> CropHealthState:
    """Collect all required data using data_agent"""
    
    print("📡 Collecting field data...")
    
    try:
        # Use our data_agent to collect comprehensive data
        data_response: DataResponse = collect_field_data(
            field_id=state['field_id'],
            lat=state['latitude'],
            lon=state['longitude'],
            radius_m=state['radius_m'],
            data_types=['satellite', 'weather']
        )
        
        if data_response.success:
            state['raw_data'] = data_response.data
            print(f"✅ Data collected: {list(data_response.data.keys())}")
            
            # Log data quality
            if 'satellite' in data_response.data:
                sat_data = data_response.data['satellite']
                if 'quality_flags' in sat_data:
                    flags = sat_data['quality_flags']
                    print(f"📊 Satellite quality: {flags}")
                    
        else:
            state['errors'].extend(data_response.errors)
            print(f"⚠️ Data collection issues: {data_response.errors}")
            
    except Exception as e:
        error_msg = f"Data ingestion failed: {str(e)}"
        state['errors'].append(error_msg)
        print(f"❌ {error_msg}")
        
    return state

def analysis_execution_node(state: CropHealthState) -> CropHealthState:
    """Execute all planned analysis tools"""
    
    print("🔬 Running analysis tools...")
    
    state['analysis_results'] = {}
    
    # Convert state to toolbox format
    toolbox_state = {
        'field_id': state['field_id'],
        'crop': state['crop'],
        'stage_das': state['das'],
        'latitude': state['latitude'],
        'longitude': state['longitude'],
        'radius_m': state['radius_m'],
        'context': {},  # Additional context if needed
        'artifacts': {}
    }
    
    # Add raw data to artifacts for toolbox compatibility
    if 'satellite' in state['raw_data']:
        sat_data = state['raw_data']['satellite']
        if 'indices' in sat_data:
            toolbox_state['artifacts']['compute_indices'] = {
                'period': {'start': '30_days_ago', 'end': 'today'},
                'image_count': sat_data.get('image_count', 0),
                'area_ha': sat_data.get('area_analyzed_ha', 0),
                **sat_data['indices']
            }
    
    if 'weather' in state['raw_data']:
        weather_data = state['raw_data']['weather']
        if 'error' not in weather_data:
            toolbox_state['artifacts']['weather'] = weather_data
    
    # Execute each tool in the plan
    for tool_name in state['analysis_plan']:
        if tool_name in AVAILABLE_TOOLS:
            try:
                print(f"⚙️ Running {tool_name}...")
                
                # Special handling for tools that need satellite data
                if tool_name == 'compute_indices' and 'satellite' in state['raw_data']:
                    # Already processed above
                    continue
                elif tool_name == 'weather_data' and 'weather' in state['raw_data']:
                    # Already processed above  
                    continue
                else:
                    # Run the tool
                    toolbox_state = AVAILABLE_TOOLS[tool_name](toolbox_state)
                
            except Exception as e:
                error_msg = f"Tool {tool_name} failed: {str(e)}"
                state['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        else:
            print(f"⚠️ Tool {tool_name} not found in toolbox")
    
    # Extract results
    state['analysis_results'] = toolbox_state['artifacts']
    
    executed_tools = len([k for k in state['analysis_results'].keys() if 'error' not in str(state['analysis_results'][k])])
    print(f"✅ Analysis complete: {executed_tools}/{len(state['analysis_plan'])} tools successful")
    
    return state

def brain_analysis_node(state: CropHealthState) -> CropHealthState:
    """LLM-powered synthesis of all analysis results"""
    
    print("🧠 Synthesizing findings with AI...")
    
    try:
        # Prepare comprehensive context for LLM
        context = {
            'field_info': {
                'id': state['field_id'],
                'crop': state['crop'],
                'days_after_sowing': state['das'],
                'location': f"{state['latitude']:.3f}, {state['longitude']:.3f}"
            },
            'satellite_analysis': state['analysis_results'].get('compute_indices', {}),
            'water_stress': state['analysis_results'].get('water_stress', {}),
            'nutrient_status': state['analysis_results'].get('nutrient_analysis', {}),
            'disease_risk': state['analysis_results'].get('disease_risk', {}),
            'weather_conditions': state['analysis_results'].get('weather', {}),
            'spatial_patterns': state['analysis_results'].get('spatial_analysis', {}),
            'yield_outlook': state['analysis_results'].get('yield_prediction', {})
        }
        
        # LLM prompt for issue identification and recommendations
        brain_prompt = f"""
You are an expert agronomist analyzing crop health data. Based on the analysis results below, identify key issues and provide specific recommendations.

FIELD DATA:
{json.dumps(context, indent=2, default=str)}

Please provide a response in valid JSON format with this exact structure:
{{
  "issues": [
    {{
      "type": "water",
      "severity": 3,
      "confidence": 0.8,
      "description": "brief description of the issue",
      "evidence": ["evidence1", "evidence2"]
    }}
  ],
  "recommendations": [
    {{
      "type": "irrigation", 
      "priority": "high",
      "action": "specific action to take",
      "timing": "this_week",
      "cost_estimate": "medium",
      "expected_benefit": "brief benefit description"
    }}
  ],
  "confidence_scores": {{
    "overall_analysis": 0.85,
    "vegetation_health": 0.9,
    "stress_detection": 0.7
  }}
}}

Important: 
- Only return valid JSON, no other text
- Use only these types: water, nutrient, disease, pest, general
- Use only these priorities: critical, high, medium, low
- Use only these timings: immediate, this_week, next_week, seasonal
- Use only these cost estimates: low, medium, high
- Severity should be 1-5
- Confidence should be 0.0-1.0
"""

        # Get LLM response
        response = llm_brain.invoke(brain_prompt)
        
        try:
            # Clean the response content - remove any non-JSON text
            content = response.content.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = content
            
            # Parse LLM response
            brain_output = json.loads(json_str)
            
            state['issues'] = brain_output.get('issues', [])
            state['recommendations'] = brain_output.get('recommendations', [])
            state['confidence_scores'] = brain_output.get('confidence_scores', {})
            
            print(f"✅ AI analysis: {len(state['issues'])} issues, {len(state['recommendations'])} recommendations")
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ LLM response parsing failed ({str(e)}), using fallback analysis")
            print(f"LLM Response: {response.content[:200]}...")  # Debug output
            state['issues'], state['recommendations'], state['confidence_scores'] = _fallback_analysis(state)
            
    except Exception as e:
        print(f"❌ Brain analysis failed: {e}")
        state['issues'], state['recommendations'], state['confidence_scores'] = _fallback_analysis(state)
    
    return state

def supervisor_node(state: CropHealthState) -> CropHealthState:
    """Apply guardrails and quality checks"""
    
    print("👮 Applying quality checks and guardrails...")
    
    # Phenology-based guardrails
    das = state['das']
    crop = state['crop'].lower()
    
    alerts = []
    
    # Early stage guardrails
    if das < 21:
        # Reduce confidence for nutrient diagnoses
        for rec in state['recommendations']:
            if rec.get('type') == 'fertilizer':
                rec['confidence'] = rec.get('confidence', 0.8) * 0.6
                rec['note'] = 'Early stage - lower confidence'
        
        alerts.append({
            'level': 'info',
            'message': 'Early growth stage - nutrient recommendations have lower confidence',
            'action': 'Validate with tissue testing'
        })
    
    # Data quality checks
    sat_results = state['analysis_results'].get('compute_indices', {})
    if sat_results.get('image_count', 0) < 1:
        alerts.append({
            'level': 'warning', 
            'message': 'Limited satellite data available',
            'action': 'Results may be less reliable'
        })
    
    # Weather data availability
    weather_data = state['analysis_results'].get('weather', {})
    if 'error' in str(weather_data):
        alerts.append({
            'level': 'warning',
            'message': 'Weather data unavailable',
            'action': 'Check local weather conditions manually'
        })
    
    # Confidence threshold enforcement
    low_confidence_items = [
        k for k, v in state['confidence_scores'].items() 
        if isinstance(v, (int, float)) and v < 0.5
    ]
    
    if low_confidence_items:
        alerts.append({
            'level': 'caution',
            'message': f'Low confidence in: {", ".join(low_confidence_items)}',
            'action': 'Consider additional data collection'
        })
    
    state['alerts'] = alerts
    
    print(f"✅ Quality checks complete: {len(alerts)} alerts generated")
    
    return state

def report_generation_node(state: CropHealthState) -> CropHealthState:
    """Generate comprehensive health report"""
    
    print("📋 Generating health report...")
    
    try:
        # Calculate overall health score
        health_score = _calculate_health_score(state)
        state['health_score'] = health_score
        
        # Determine status
        if health_score >= 80:
            health_status = "Excellent"
            status_emoji = "🟢"
        elif health_score >= 65:
            health_status = "Good"
            status_emoji = "🟡"
        elif health_score >= 45:
            health_status = "Attention Needed"
            status_emoji = "🟠"
        else:
            health_status = "Critical"
            status_emoji = "🔴"
        
        # Get key metrics
        sat_data = state['analysis_results'].get('compute_indices', {})
        ndvi_mean = sat_data.get('NDVI', {}).get('mean', 0) if isinstance(sat_data.get('NDVI'), dict) else 0
        
        # Processing time
        start_time = datetime.fromisoformat(state['started_at'])
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        state['finished_at'] = end_time.isoformat()
        state['processing_time'] = processing_time
        state['status'] = 'completed'
        
        # Store report data directly in state
        state['health_status'] = health_status
        state['status_emoji'] = status_emoji
        state['ndvi_mean'] = round(ndvi_mean, 3)
        state['issues_found'] = len(state['issues'])
        state['recommendations_count'] = len(state['recommendations'])
        state['alerts_count'] = len(state['alerts'])
        state['tools_executed'] = len(state['analysis_results'])
        
        print(f"✅ Report generated: {health_status} ({health_score}/100)")
        
    except Exception as e:
        error_msg = f"Report generation failed: {str(e)}"
        state['errors'].append(error_msg)
        print(f"❌ {error_msg}")
    
    return state

# =========================
# HELPER FUNCTIONS
# =========================

def _fallback_analysis(state: CropHealthState) -> tuple:
    """Fallback analysis when LLM fails"""
    
    issues = []
    recommendations = []
    confidence = {"overall_analysis": 0.6, "vegetation_health": 0.7, "stress_detection": 0.5}
    
    # Basic rule-based analysis
    sat_data = state['analysis_results'].get('compute_indices', {})
    
    # Check NDVI values
    if isinstance(sat_data.get('NDVI'), dict):
        ndvi = sat_data['NDVI'].get('mean', 0)
        
        if ndvi < 0.4:
            issues.append({
                'type': 'general',
                'severity': 4,
                'confidence': 0.8,
                'description': 'Low vegetation health detected',
                'evidence': [f'NDVI = {ndvi:.3f}']
            })
            
            recommendations.append({
                'type': 'monitoring',
                'priority': 'high',
                'action': 'Conduct field inspection to identify stress causes',
                'timing': 'this_week',
                'cost_estimate': 'low',
                'expected_benefit': 'Identify and address stress factors'
            })
        elif ndvi < 0.6:
            issues.append({
                'type': 'general',
                'severity': 2,
                'confidence': 0.7,
                'description': 'Moderate vegetation vigor detected',
                'evidence': [f'NDVI = {ndvi:.3f}']
            })
            
            recommendations.append({
                'type': 'monitoring',
                'priority': 'medium',
                'action': 'Monitor crop development closely',
                'timing': 'next_week',
                'cost_estimate': 'low',
                'expected_benefit': 'Early detection of potential issues'
            })
    
    # Check weather-based stress
    weather_data = state['analysis_results'].get('weather', {})
    if isinstance(weather_data, dict) and 'precipitation' in weather_data:
        recent_precip = weather_data.get('precipitation_7day', 0)
        if recent_precip < 10:  # Less than 10mm in 7 days
            issues.append({
                'type': 'water',
                'severity': 3,
                'confidence': 0.8,
                'description': 'Low recent precipitation detected',
                'evidence': [f'7-day precipitation: {recent_precip}mm']
            })
            
            recommendations.append({
                'type': 'irrigation',
                'priority': 'medium',
                'action': 'Consider supplemental irrigation',
                'timing': 'this_week',
                'cost_estimate': 'medium',
                'expected_benefit': 'Maintain adequate soil moisture'
            })
    
    return issues, recommendations, confidence

def _calculate_health_score(state: CropHealthState) -> int:
    """Calculate overall health score based on analysis results"""
    
    base_score = 80
    
    # NDVI-based adjustment
    sat_data = state['analysis_results'].get('compute_indices', {})
    if isinstance(sat_data.get('NDVI'), dict):
        ndvi = sat_data['NDVI'].get('mean', 0.5)
        
        if ndvi < 0.3:
            base_score -= 30
        elif ndvi < 0.5:
            base_score -= 15
        elif ndvi > 0.8:
            base_score += 10
    
    # Issue-based deductions
    for issue in state['issues']:
        severity = issue.get('severity', 1)
        base_score -= severity * 5
    
    # Alert-based adjustments
    critical_alerts = len([a for a in state['alerts'] if a.get('level') == 'warning'])
    base_score -= critical_alerts * 3
    
    return max(0, min(100, base_score))

# =========================
# LANGGRAPH SETUP
# =========================

def create_crop_health_agent():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(CropHealthState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("data_ingestion", data_ingestion_node)
    workflow.add_node("analysis_execution", analysis_execution_node)
    workflow.add_node("brain_analysis", brain_analysis_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("report_generation", report_generation_node)
    
    # Define workflow
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "data_ingestion")
    workflow.add_edge("data_ingestion", "analysis_execution")
    workflow.add_edge("analysis_execution", "brain_analysis")
    workflow.add_edge("brain_analysis", "supervisor")
    workflow.add_edge("supervisor", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow.compile()

# =========================
# PUBLIC INTERFACE
# =========================

def analyze_field(field_id: str, crop: str, latitude: float, longitude: float, 
                 das: int, radius_m: int = 1000) -> Dict[str, Any]:
    """
    Main function to analyze a field's health
    """
    
    print(f"\n🌱 Starting Crop Health Analysis")
    print(f"{'='*50}")
    print(f"Field: {field_id}")
    print(f"Crop: {crop.title()} ({das} DAS)")
    print(f"Location: {latitude:.4f}, {longitude:.4f}")
    print(f"{'='*50}")
    
    # Create initial state
    initial_state = CropHealthState(
        field_id=field_id,
        crop=crop,
        latitude=latitude,
        longitude=longitude,
        radius_m=radius_m,
        das=das,
        raw_data={},
        analysis_results={},
        issues=[],
        recommendations=[],
        alerts=[],
        analysis_plan=[],
        confidence_scores={},
        health_score=0,
        status='initialized',
        # Add the new fields with defaults
        health_status=None,
        status_emoji=None,
        ndvi_mean=None,
        issues_found=None,
        recommendations_count=None,
        alerts_count=None,
        tools_executed=None,
        started_at='',
        finished_at=None,
        processing_time=None,
        errors=[]
    )
    
    # Create and run agent
    agent = create_crop_health_agent()
    
    try:
        # Execute workflow
        final_state = agent.invoke(initial_state)
        
        # Print summary
        if final_state['status'] == 'completed':
            print(f"\n🎉 Analysis Complete!")
            print(f"{'='*50}")
            print(f"{final_state.get('status_emoji', '📊')} Health Status: {final_state.get('health_status', 'Unknown')} ({final_state.get('health_score', 0)}/100)")
            print(f"📈 NDVI: {final_state.get('ndvi_mean', 'N/A')}")
            print(f"⚠️  Issues Found: {final_state.get('issues_found', 0)}")
            print(f"💡 Recommendations: {final_state.get('recommendations_count', 0)}")
            print(f"⚡ Processing Time: {final_state.get('processing_time', 0):.1f}s")
            
            if final_state['errors']:
                print(f"⚠️  Warnings: {len(final_state['errors'])}")
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        return {'error': str(e), 'status': 'failed'}
    
    
if __name__ == "__main__":
    # Test the complete agent
    result = analyze_field(
        field_id="Iowa_Test_Field_v2",
        crop="corn",
        latitude=41.878003,
        longitude=-93.097702,
        das=45,
        radius_m=1000
    )
    
    # Print detailed results
    if result.get('status') == 'completed':
        print(f"\n📋 DETAILED RESULTS:")
        print(f"Issues: {len(result.get('issues', []))}")
        for issue in result.get('issues', [])[:3]:  # Show top 3
            print(f"  - {issue.get('description', 'N/A')} (Severity: {issue.get('severity', 'N/A')})")
        
        print(f"Recommendations: {len(result.get('recommendations', []))}")
        for rec in result.get('recommendations', [])[:3]:  # Show top 3
            print(f"  - {rec.get('action', 'N/A')} ({rec.get('priority', 'N/A')} priority)")