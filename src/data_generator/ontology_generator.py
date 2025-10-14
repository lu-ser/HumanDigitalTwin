"""
Generatore di dati IoT basato sull'ontologia onto.owl.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import random
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS


class OntologyDataGenerator:
    """
    Genera dati IoT realistici basati sull'ontologia.
    Analizza onto.owl per capire quali sensori/dispositivi esistono e genera dati conformi.
    """

    def __init__(self, ontology_path: str = None):
        """
        Inizializza il generatore.

        Args:
            ontology_path: Percorso al file onto.owl
        """
        if ontology_path is None:
            # Percorso default
            project_root = Path(__file__).parent.parent.parent
            ontology_path = project_root / "ontologies" / "onto.owl"

        self.ontology_path = Path(ontology_path)
        self.graph = Graph()

        # Carica l'ontologia
        if self.ontology_path.exists():
            self.graph.parse(str(self.ontology_path))

        # Namespace dell'ontologia
        self.ns = Namespace("http://www.semanticweb.org/jbagwell/ontologies/2017/9/untitled-ontology-6#")

        # Definisci i device profiles basati sull'ontologia
        self._init_device_profiles()

    def _init_device_profiles(self):
        """Inizializza i profili dei dispositivi basati sull'ontologia."""

        # Fitbit - focus su activity tracking e sleep
        self.device_profiles = {
            "fitbit": {
                "device_type": "fitbit",
                "sensors": ["heartRateMonitor", "3axisAccelerometer", "altimeter", "gps"],
                "metrics": {
                    # Steps e attività
                    "steps": (0, 20000),
                    "minutesSedentary": (200, 800),
                    "minutesLightlyActive": (50, 200),
                    "minutesFairlyActive": (10, 60),
                    "minutesVeryActive": (0, 60),
                    "floors": (0, 30),
                    "distanceMiles": (0.0, 15.0),
                    "elevation": (0, 500),
                    "calories": (1500, 3500),

                    # Sleep
                    "minutesAsleep": (300, 540),
                    "minutesAwake": (5, 60),
                    "awakeningsCount": (0, 10),
                    "minutesToFallAsleep": (5, 30),
                    "minutesAfterWakeup": (5, 30),
                    "timeInBed": (360, 600),

                    # User data
                    "weight": (50.0, 120.0),
                    "height": (150.0, 200.0),
                    "bmi": (18.0, 35.0),
                    "fat": (10.0, 40.0),
                }
            },
            "garmin": {
                "device_type": "garmin",
                "sensors": ["heartRateMonitor", "gps", "altimeter"],
                "metrics": {
                    # Heart rate
                    "averageHeartRate": (60, 150),
                    "maximunHeartRate": (120, 200),

                    # Activity
                    "activityName": None,  # Stringa
                    "activityType": None,  # Stringa
                    "duration": (300, 7200),  # secondi
                    "distanceMiles": (0.0, 26.0),
                    "calories": (100, 2000),
                    "averageSpeedPace": (5.0, 15.0),  # mph
                    "maxSpeedBestPace": (8.0, 20.0),

                    # Advanced metrics
                    "efficiency": (0.5, 1.0),
                    "estamatedIntensityFactor": (0.5, 1.2),
                    "estimatedTrainingStessScore": (0, 300),
                }
            },
            "jawbone": {
                "device_type": "jawbone",
                "sensors": ["3axisAccelerometer", "3axisGyroscope"],
                "metrics": {
                    # Activity
                    "steps": (0, 20000),
                    "activeTime": (0, 300),  # minuti
                    "activeTimeSeconds": (0, 18000),
                    "distanceKilometers": (0.0, 20.0),
                    "calories": (1500, 3500),
                    "inactive": (300, 1200),  # minuti sedentario
                    "percentActive": (10.0, 80.0),
                }
            }
        }

        # Activity types per Garmin
        self.activity_types = [
            "running", "cycling", "walking", "swimming", "gym",
            "hiking", "yoga", "tennis", "soccer"
        ]

    def generate_data(
        self,
        device_type: str,
        device_id: Optional[str] = None,
        num_records: int = 1,
        time_interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Genera dati IoT per un dispositivo specifico.

        Args:
            device_type: Tipo di dispositivo ('fitbit', 'garmin', 'jawbone')
            device_id: ID del dispositivo (se None, genera automaticamente)
            num_records: Numero di record da generare
            time_interval_minutes: Intervallo tra i record in minuti

        Returns:
            Lista di dizionari con i dati generati

        Raises:
            ValueError: Se device_type non è supportato
        """
        if device_type not in self.device_profiles:
            available = ', '.join(self.device_profiles.keys())
            raise ValueError(f"Device type '{device_type}' non supportato. Disponibili: {available}")

        profile = self.device_profiles[device_type]

        if device_id is None:
            device_id = f"{device_type}_{random.randint(1000, 9999)}"

        records = []
        base_time = datetime.now()

        for i in range(num_records):
            timestamp = base_time - timedelta(minutes=i * time_interval_minutes)

            # Genera i dati per tutte le metriche
            data = {}
            for metric, value_range in profile["metrics"].items():
                if value_range is None:
                    # Metriche string (es: activityType)
                    if metric == "activityType":
                        data[metric] = random.choice(self.activity_types)
                    elif metric == "activityName":
                        data[metric] = f"Activity_{random.randint(1, 100)}"
                else:
                    # Metriche numeriche
                    min_val, max_val = value_range
                    if isinstance(min_val, float):
                        data[metric] = round(random.uniform(min_val, max_val), 2)
                    else:
                        data[metric] = random.randint(min_val, max_val)

            record = {
                "device_type": device_type,
                "device_id": device_id,
                "timestamp": timestamp.isoformat(),
                "data": data,
                "metadata": {
                    "sensors": profile["sensors"],
                    "generated": True,
                    "ontology": "onto.owl"
                }
            }

            records.append(record)

        return records

    def get_available_devices(self) -> List[str]:
        """
        Restituisce la lista dei dispositivi disponibili.

        Returns:
            Lista di nomi dei dispositivi
        """
        return list(self.device_profiles.keys())

    def get_device_metrics(self, device_type: str) -> Dict[str, Any]:
        """
        Restituisce le metriche disponibili per un dispositivo.

        Args:
            device_type: Tipo di dispositivo

        Returns:
            Dizionario con le metriche e i loro range

        Raises:
            ValueError: Se device_type non esiste
        """
        if device_type not in self.device_profiles:
            raise ValueError(f"Device type '{device_type}' non trovato")

        profile = self.device_profiles[device_type]
        return {
            "device_type": device_type,
            "sensors": profile["sensors"],
            "metrics": profile["metrics"]
        }

    def generate_realistic_day(self, device_type: str, device_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Genera una giornata completa di dati realistici (24 campioni, uno all'ora).

        Args:
            device_type: Tipo di dispositivo
            device_id: ID del dispositivo

        Returns:
            Lista di 24 record, uno per ogni ora
        """
        return self.generate_data(
            device_type=device_type,
            device_id=device_id,
            num_records=24,
            time_interval_minutes=60
        )

    def generate_sample_for_all_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Genera dati di esempio per tutti i dispositivi disponibili.

        Returns:
            Dizionario con device_type come chiave e lista di record come valore
        """
        samples = {}
        for device_type in self.device_profiles.keys():
            samples[device_type] = self.generate_data(
                device_type=device_type,
                num_records=5
            )
        return samples
