"""Command-line interface for DAPAC."""

import argparse
import json
import sys
import cv2
from pathlib import Path

from .pipeline import AdPlacementPipeline


def analyze_command(args):
    """Run analyze mode: find candidate placements."""
    poster_path = Path(args.input)
    if not poster_path.exists():
        print(f"Error: Poster not found: {poster_path}")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    pipeline = AdPlacementPipeline(device=args.device, verbose=args.verbose)
    analysis = pipeline.analyze(str(poster_path))
    
    # Save results JSON
    results = {
        'input': str(poster_path),
        'candidates': [
            {
                'id': i,
                'x': c.x,
                'y': c.y,
                'w': c.w,
                'h': c.h,
                'score': float(c.score),
                'reasons': c.reasons,
            }
            for i, c in enumerate(analysis['candidates'])
        ],
        'timing': {k: float(v) for k, v in analysis['timing'].items()},
    }
    
    results_path = output_dir / f"{poster_path.stem}_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Analysis complete")
    print(f"  Total time: {analysis['timing']['total']:.2f}s")
    print(f"  Candidates found: {len(analysis['candidates'])}")
    print(f"  Results saved to: {results_path}")
    
    # Save overlay images
    if args.save_overlays:
        # Protected mask
        mask_path = output_dir / f"{poster_path.stem}_protected_mask.png"
        cv2.imwrite(str(mask_path), analysis['protected_mask'])
        
        # Saliency map
        sal_path = output_dir / f"{poster_path.stem}_saliency.png"
        sal_vis = (analysis['saliency_map'] * 255).astype('uint8')
        sal_vis = cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(sal_path), sal_vis)
        
        # Candidates overlay
        overlay = analysis['image'].copy()
        for i, cand in enumerate(analysis['candidates'][:5]):
            color = (0, 255, 0) if i == 0 else (255, 200, 0)
            cv2.rectangle(overlay, (cand.x, cand.y), (cand.x + cand.w, cand.y + cand.h), color, 2)
            cv2.putText(overlay, f"#{i+1}", (cand.x + 5, cand.y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        overlay_path = output_dir / f"{poster_path.stem}_candidates.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        
        print(f"  Overlays saved to: {output_dir}")
    
    return 0


def compose_command(args):
    """Run compose mode: composite ad onto selected candidate."""
    poster_path = Path(args.input)
    ad_path = Path(args.ad)
    
    if not poster_path.exists():
        print(f"Error: Poster not found: {poster_path}")
        return 1
    if not ad_path.exists():
        print(f"Error: Ad asset not found: {ad_path}")
        return 1
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run analysis first
    pipeline = AdPlacementPipeline(device=args.device, verbose=args.verbose)
    analysis = pipeline.analyze(str(poster_path))
    
    # Select candidate
    candidate_id = args.candidate
    if candidate_id < 0 or candidate_id >= len(analysis['candidates']):
        print(f"Error: Invalid candidate ID {candidate_id}. Valid range: 0-{len(analysis['candidates'])-1}")
        return 1
    
    selected_candidate = analysis['candidates'][candidate_id]
    
    print(f"\nCompositing ad onto candidate #{candidate_id}:")
    print(f"  Position: (x={selected_candidate.x}, y={selected_candidate.y})")
    print(f"  Size: {selected_candidate.w}x{selected_candidate.h}")
    print(f"  Score: {selected_candidate.score:.3f}")
    
    # Composite
    result = pipeline.compose(
        analysis['image'],
        str(ad_path),
        selected_candidate,
        analysis['protected_mask']
    )
    
    if not result['valid']:
        print(f"\n✗ Composite failed: {result['warnings']}")
        return 1
    
    # Save
    cv2.imwrite(str(output_path), result['composite'])
    print(f"\n✓ Composite successful")
    if result['warnings']:
        print(f"  Warnings: {result['warnings']}")
    print(f"  Saved to: {output_path}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DAPAC - Discreet Ad Placement & Auto-Compositing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze poster and find candidate placements')
    analyze_parser.add_argument('--input', '-i', required=True, help='Input poster image path')
    analyze_parser.add_argument('--output', '-o', default='outputs/', help='Output directory')
    analyze_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    analyze_parser.add_argument('--save-overlays', action='store_true', help='Save visualization overlays')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Compose command
    compose_parser = subparsers.add_parser('compose', help='Composite ad onto poster')
    compose_parser.add_argument('--input', '-i', required=True, help='Input poster image path')
    compose_parser.add_argument('--ad', '-a', required=True, help='Ad asset image path')
    compose_parser.add_argument('--candidate', '-c', type=int, default=0, help='Candidate ID to use (default: 0=best)')
    compose_parser.add_argument('--output', '-o', required=True, help='Output composite image path')
    compose_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    compose_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'analyze':
        return analyze_command(args)
    elif args.command == 'compose':
        return compose_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
