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
        
        # Access the actual fields from the state
        print(f"Field: {result.get('field_id', 'N/A')}")
        print(f"Crop: {result.get('crop', 'N/A').title()} ({result.get('das', 'N/A')} DAS)")
        print(f"Location: {result.get('latitude', 'N/A'):.4f}, {result.get('longitude', 'N/A'):.4f}")
        print(f"Health Score: {result.get('health_score', 'N/A')}/100")
        print(f"Status: {result.get('status_emoji', '')} {result.get('health_status', 'N/A')}")
        print(f"NDVI: {result.get('ndvi_mean', 'N/A')}")
        print(f"Processing Time: {result.get('processing_time', 0):.1f}s")
        
        # Display issues
        issues = result.get('issues', [])
        print(f"\nIssues Found ({len(issues)}):")
        if issues:
            for i, issue in enumerate(issues[:5], 1):  # Show top 5
                severity = issue.get('severity', 'N/A')
                confidence = issue.get('confidence', 'N/A')
                description = issue.get('description', 'N/A')
                print(f"  {i}. {description}")
                print(f"     Severity: {severity}/5, Confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else f"     Severity: {severity}/5")
                
                # Show evidence if available
                evidence = issue.get('evidence', [])
                if evidence:
                    print(f"     Evidence: {', '.join(evidence[:2])}")  # Show first 2 pieces of evidence
        else:
            print("  ✅ No issues detected")
        
        # Display recommendations  
        recommendations = result.get('recommendations', [])
        print(f"\nRecommendations ({len(recommendations)}):")
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                action = rec.get('action', 'N/A')
                priority = rec.get('priority', 'N/A')
                timing = rec.get('timing', 'N/A')
                rec_type = rec.get('type', 'N/A')
                
                print(f"  {i}. {action}")
                print(f"     Priority: {priority}, Type: {rec_type}, Timing: {timing}")
                
                # Show expected benefit if available
                benefit = rec.get('expected_benefit')
                if benefit:
                    print(f"     Expected Benefit: {benefit}")
        else:
            print("  ℹ️ No specific recommendations needed")
        
        # Display alerts
        alerts = result.get('alerts', [])
        if alerts:
            print(f"\nAlerts & Warnings ({len(alerts)}):")
            for i, alert in enumerate(alerts, 1):
                level = alert.get('level', 'info')
                message = alert.get('message', 'N/A')
                action = alert.get('action', '')
                
                level_emoji = {'info': 'ℹ️', 'warning': '⚠️', 'caution': '⚡'}.get(level, '📝')
                print(f"  {i}. {level_emoji} {message}")
                if action:
                    print(f"     Action: {action}")
        
        # Display confidence scores
        confidence_scores = result.get('confidence_scores', {})
        if confidence_scores:
            print(f"\nConfidence Scores:")
            for key, score in confidence_scores.items():
                if isinstance(score, (int, float)):
                    print(f"  {key.replace('_', ' ').title()}: {score:.2f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {score}")
        
        # Display analysis results summary
        analysis_results = result.get('analysis_results', {})
        if analysis_results:
            print(f"\nAnalysis Tools Executed ({len(analysis_results)}):")
            for tool_name, tool_result in analysis_results.items():
                # Check if tool executed successfully (no error in result)
                if 'error' not in str(tool_result).lower():
                    print(f"  ✅ {tool_name.replace('_', ' ').title()}")
                else:
                    print(f"  ❌ {tool_name.replace('_', ' ').title()} (failed)")
        
        # Display errors if any
        errors = result.get('errors', [])
        if errors:
            print(f"\nErrors/Warnings ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
                
    else:
        print(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Show any errors that occurred
        errors = result.get('errors', [])
        if errors:
            print("Errors encountered:")
            for error in errors:
                print(f"  - {error}")