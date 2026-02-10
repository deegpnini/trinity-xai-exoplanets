"""
Handoff Protocol - JSON Communication Between Nodes
Part of Gemini/Dola vectors - Seamless node communication

This module implements the standardized handoff protocol for transferring
data and context between different processing nodes.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json


class HandoffStatus(Enum):
    """Status of a handoff operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HandoffMetadata:
    """Metadata for handoff operations."""
    source_node: str
    target_node: str
    timestamp: float
    priority: int
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class HandoffPayload:
    """Payload structure for handoffs."""
    data: Any
    data_type: str
    encoding: str = "utf-8"
    compressed: bool = False
    checksum: Optional[str] = None


class HandoffProtocol:
    """
    Standardized protocol for inter-node communication.
    
    Ensures reliable data transfer between Sensorial (A70) and
    Cognitive (Raspberry Pi) nodes.
    """
    
    PROTOCOL_VERSION = "1.0"
    
    def __init__(self):
        self.handoff_log: List[Dict[str, Any]] = []
        self.active_handoffs: Dict[str, HandoffStatus] = {}
        
    def create_handoff(
        self,
        source: str,
        target: str,
        payload: Any,
        priority: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized handoff package.
        
        Args:
            source: Source node identifier
            target: Target node identifier
            payload: Data to transfer
            priority: Priority level (1=low, 5=critical)
            context: Additional context information
            
        Returns:
            Complete handoff package
        """
        import time
        
        handoff_id = self._generate_handoff_id(source, target)
        
        metadata = HandoffMetadata(
            source_node=source,
            target_node=target,
            timestamp=time.time(),
            priority=priority
        )
        
        payload_obj = HandoffPayload(
            data=payload,
            data_type=type(payload).__name__,
            checksum=self._calculate_checksum(payload)
        )
        
        handoff_package = {
            'handoff_id': handoff_id,
            'protocol_version': self.PROTOCOL_VERSION,
            'metadata': asdict(metadata),
            'payload': asdict(payload_obj),
            'context': context or {},
            'status': HandoffStatus.PENDING.value
        }
        
        self.active_handoffs[handoff_id] = HandoffStatus.PENDING
        self._log_handoff(handoff_package)
        
        return handoff_package
    
    def validate_handoff(self, handoff_package: Dict[str, Any]) -> bool:
        """
        Validate handoff package integrity.
        
        Args:
            handoff_package: Package to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['handoff_id', 'protocol_version', 'metadata', 'payload']
        
        # Check required fields
        if not all(field in handoff_package for field in required_fields):
            return False
        
        # Validate protocol version
        if handoff_package['protocol_version'] != self.PROTOCOL_VERSION:
            return False
        
        # Validate checksum if present
        payload = handoff_package['payload']
        if payload.get('checksum'):
            calculated = self._calculate_checksum(payload['data'])
            if calculated != payload['checksum']:
                return False
        
        return True
    
    def process_handoff(self, handoff_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming handoff.
        
        Args:
            handoff_package: Package to process
            
        Returns:
            Processing result with status
        """
        handoff_id = handoff_package['handoff_id']
        
        # Validate first
        if not self.validate_handoff(handoff_package):
            self.active_handoffs[handoff_id] = HandoffStatus.FAILED
            return {
                'handoff_id': handoff_id,
                'status': HandoffStatus.FAILED.value,
                'error': 'Validation failed'
            }
        
        # Update status
        self.active_handoffs[handoff_id] = HandoffStatus.IN_PROGRESS
        
        # Extract and process payload
        payload_data = handoff_package['payload']['data']
        
        # Process based on data type
        result = self._process_payload(payload_data, handoff_package['context'])
        
        # Mark completed
        self.active_handoffs[handoff_id] = HandoffStatus.COMPLETED
        
        return {
            'handoff_id': handoff_id,
            'status': HandoffStatus.COMPLETED.value,
            'result': result
        }
    
    def _generate_handoff_id(self, source: str, target: str) -> str:
        """Generate unique handoff ID."""
        import time
        timestamp = int(time.time() * 1000)
        return f"{source}_{target}_{timestamp}"
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity."""
        import hashlib
        data_str = str(data).encode('utf-8')
        return hashlib.md5(data_str).hexdigest()
    
    def _process_payload(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process the payload data."""
        # Placeholder for actual processing logic
        # In production, this would route to appropriate handlers
        return {
            'processed': True,
            'data': data,
            'context_applied': bool(context)
        }
    
    def _log_handoff(self, handoff_package: Dict[str, Any]):
        """Log handoff for audit trail."""
        self.handoff_log.append({
            'handoff_id': handoff_package['handoff_id'],
            'source': handoff_package['metadata']['source_node'],
            'target': handoff_package['metadata']['target_node'],
            'timestamp': handoff_package['metadata']['timestamp'],
            'status': handoff_package['status']
        })
    
    def get_handoff_status(self, handoff_id: str) -> Optional[HandoffStatus]:
        """Get status of a specific handoff."""
        return self.active_handoffs.get(handoff_id)
    
    def serialize_handoff(self, handoff_package: Dict[str, Any]) -> str:
        """Serialize handoff to JSON string."""
        return json.dumps(handoff_package, indent=2)
    
    def deserialize_handoff(self, json_str: str) -> Dict[str, Any]:
        """Deserialize handoff from JSON string."""
        return json.loads(json_str)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get handoff statistics."""
        total = len(self.handoff_log)
        if total == 0:
            return {'total': 0, 'success_rate': 0.0}
        
        completed = sum(
            1 for h_id, status in self.active_handoffs.items()
            if status == HandoffStatus.COMPLETED
        )
        
        return {
            'total_handoffs': total,
            'completed': completed,
            'success_rate': completed / total if total > 0 else 0.0,
            'average_per_minute': 0.0  # Would be calculated from timestamps
        }


# Example usage
if __name__ == "__main__":
    protocol = HandoffProtocol()
    
    # Create a handoff from sensorial to cognitive
    handoff = protocol.create_handoff(
        source="galaxy_a70_sensorial",
        target="rpi5_cognitive",
        payload={"transcription": "Hello world", "emotion": "happy"},
        priority=3,
        context={"child_age": 8, "supervised": True}
    )
    
    print(f"Created handoff: {handoff['handoff_id']}")
    print(f"Status: {handoff['status']}")
    
    # Serialize for network transfer
    json_handoff = protocol.serialize_handoff(handoff)
    print(f"\nSerialized length: {len(json_handoff)} bytes")
    
    # Process the handoff
    result = protocol.process_handoff(handoff)
    print(f"\nProcessing result: {result['status']}")
    
    # Get statistics
    stats = protocol.get_statistics()
    print(f"\nStatistics: {stats}")
