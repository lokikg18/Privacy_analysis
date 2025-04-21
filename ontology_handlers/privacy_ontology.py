from pathlib import Path
from typing import Dict, List, Optional

from owlready2 import *

from models.domain_models import (ComplianceStatus, DataType, PrivacyPolicy,
                                  PrivacyRisk, PrivacyRiskLevel)


class PrivacyOntologyHandler:
    def __init__(self, ontology_path: str = "ontologies/privacy_ontology.owl"):
        """Initialize the Privacy Ontology Handler.

        Args:
            ontology_path (str): Path to the OWL ontology file
        """
        self.onto_path = Path(ontology_path)
        self.world = World()
        self.ontology = self.world.get_ontology(
            self.onto_path.absolute().as_uri()
        ).load()

    def get_personal_data_types(self) -> List[str]:
        """Get all personal data types defined in the ontology."""
        personal_data = self.ontology.search(type=self.ontology.PersonalData)
        return [data.dataType for data in personal_data if hasattr(data, "dataType")]

    def get_risk_levels(self) -> Dict[str, int]:
        """Get all risks and their levels defined in the ontology."""
        risks = self.ontology.search(type=self.ontology.Risk)
        return {
            str(risk): risk.riskLevel for risk in risks if hasattr(risk, "riskLevel")
        }

    def get_mitigation_strategies(self, risk_level: int) -> List[str]:
        """Get mitigation strategies for a given risk level.

        Args:
            risk_level (int): The risk level (1-5)

        Returns:
            List[str]: List of mitigation strategies
        """
        if not 1 <= risk_level <= 5:
            raise ValueError("Risk level must be between 1 and 5")

        mitigation_strategies = []

        # Get all mitigation strategies for the given risk level
        strategies = self.ontology.search(
            type=self.ontology.MitigationStrategy, riskLevel=risk_level
        )

        for strategy in strategies:
            if hasattr(strategy, "description"):
                mitigation_strategies.append(strategy.description)

        return mitigation_strategies

    def get_risk(self, risk_id: str) -> Optional[PrivacyRisk]:
        """Get a privacy risk by ID.

        Args:
            risk_id (str): The ID of the risk to retrieve

        Returns:
            Optional[PrivacyRisk]: The privacy risk if found, None otherwise
        """
        risks = self.ontology.search(type=self.ontology.Risk, id=risk_id)

        if not risks:
            return None

        risk = risks[0]
        return PrivacyRisk(
            id=risk.id,
            level=risk.riskLevel,
            description=risk.description,
            affected_data_types=[DataType(dt) for dt in risk.affectedDataTypes],
            affected_devices=risk.affectedDevices,
            mitigation_strategy=risk.mitigationStrategy,
            detected_at=risk.detectedAt,
            resolved_at=risk.resolvedAt,
            status=risk.status,
        )

    def get_risks(self, status: Optional[str] = None) -> List[PrivacyRisk]:
        """Get all privacy risks, optionally filtered by status.

        Args:
            status (Optional[str]): Filter risks by status (active, mitigated, resolved)

        Returns:
            List[PrivacyRisk]: List of privacy risks
        """
        query = {"type": self.ontology.Risk}
        if status:
            query["status"] = status

        risks = self.ontology.search(**query)

        return [
            PrivacyRisk(
                id=risk.id,
                level=risk.riskLevel,
                description=risk.description,
                affected_data_types=[DataType(dt) for dt in risk.affectedDataTypes],
                affected_devices=risk.affectedDevices,
                mitigation_strategy=risk.mitigationStrategy,
                detected_at=risk.detectedAt,
                resolved_at=risk.resolvedAt,
                status=risk.status,
            )
            for risk in risks
        ]

    def add_policy(self, policy: PrivacyPolicy) -> None:
        """Add a new privacy policy to the ontology.

        Args:
            policy (PrivacyPolicy): The privacy policy to add
        """
        with self.ontology:
            new_policy = self.ontology.PrivacyPolicy()
            new_policy.id = policy.id
            new_policy.name = policy.name
            new_policy.description = policy.description
            new_policy.dataTypes = [dt.value for dt in policy.data_types]
            new_policy.retentionPeriod = policy.retention_period
            new_policy.accessControlRules = policy.access_control_rules
            new_policy.createdAt = policy.created_at
            new_policy.updatedAt = policy.updated_at

    def get_policy(self, policy_id: str) -> Optional[PrivacyPolicy]:
        """Get a privacy policy by ID.

        Args:
            policy_id (str): The ID of the policy to retrieve

        Returns:
            Optional[PrivacyPolicy]: The privacy policy if found, None otherwise
        """
        policies = self.ontology.search(type=self.ontology.PrivacyPolicy, id=policy_id)

        if not policies:
            return None

        policy = policies[0]
        return PrivacyPolicy(
            id=policy.id,
            name=policy.name,
            description=policy.description,
            data_types=[DataType(dt) for dt in policy.dataTypes],
            retention_period=policy.retentionPeriod,
            access_control_rules=policy.accessControlRules,
            created_at=policy.createdAt,
            updated_at=policy.updatedAt,
        )

    def add_personal_data(self, data_type: str) -> None:
        """Add a new personal data type to the ontology.

        Args:
            data_type (str): Type of personal data
        """
        with self.ontology:
            new_data = self.ontology.PersonalData()
            new_data.dataType = data_type

    def add_risk(self, risk_name: str, risk_level: int) -> None:
        """Add a new risk to the ontology.

        Args:
            risk_name (str): Name of the risk
            risk_level (int): Level of the risk (1-5)
        """
        if not 1 <= risk_level <= 5:
            raise ValueError("Risk level must be between 1 and 5")

        with self.ontology:
            new_risk = self.ontology.Risk()
            new_risk.riskLevel = risk_level

    def save_ontology(self) -> None:
        """Save the ontology to file."""
        self.ontology.save(file=str(self.onto_path), format="rdfxml")


if __name__ == "__main__":
    # Example usage
    handler = PrivacyOntologyHandler()

    # Add some example data
    handler.add_personal_data("location")
    handler.add_personal_data("health_data")
    handler.add_risk("data_breach", 5)
    handler.add_risk("unauthorized_access", 4)

    # Save changes
    handler.save_ontology()

    # Print current data
    print("Personal Data Types:", handler.get_personal_data_types())
    print("Risks and Levels:", handler.get_risk_levels())
