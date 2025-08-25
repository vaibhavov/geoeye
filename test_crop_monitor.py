import sys
sys.path.append('src/agents')

from crop_health_monitor2 import analyze_field

# Test the complete monitoring workflow
if __name__ == "__main__":
    result = analyze_field(
        field_id="Test_Farm_001",
        crop="corn",
        latitude=41.878003,
        longitude=-93.097702,
        das=60,  # Mid-season corn
        radius_m=1500
    )
    
    print(f"\nFinal Status: {result.get('status', 'unknown')}")
    
    if result.get('status') == 'completed':
        print("\n" + "="*60)
        print("CROP HEALTH MONITORING REPORT")
        print("="*60)
        
        summary = result.get('report_summary', {})
        print(f"Field: {summary.get('field_id', 'N/A')}")
        print(f"Health Score: {summary.get('health_summary', {}).get('overall_score', 'N/A')}/100")
        print(f"Status: {summary.get('health_summary', {}).get('status', 'N/A')}")
        
        print(f"\nKey Findings:")
        for issue in result.get('issues', []):
            print(f"• {issue.get('description', 'N/A')} (Severity: {issue.get('severity', 'N/A')}/5)")
        
        print(f"\nRecommendations:")
        for rec in result.get('recommendations', []):
            print(f"• {rec.get('action', 'N/A')} - {rec.get('priority', 'N/A')} priority")
    else:
        print(f"Analysis failed: {result.get('error', 'Unknown error')}")