from sqlalchemy import create_engine, Column, DateTime, Enum, ForeignKey, Integer, String, func, Float
from sqlalchemy.orm import Mapped, relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pathlib import Path
from typing import List
from itsper.types import ItsperAnnotationTypes
from rich.console import Console
from rich.table import Table
from itsper.io import get_logger


Base = declarative_base()
console = Console()
logger = get_logger(__name__)


class ItsperManifest(Base):
    __tablename__ = "manifest"

    id = Column(Integer, primary_key=True)
    created = Column(DateTime(timezone=True), default=func.now())
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    name = Column(String, unique=True)

    patients = relationship("Patient", back_populates="manifest")


class Annotation(Base):
    __tablename__ = "annotations"

    filename = Column(String, nullable=False)
    id = Column(Integer, primary_key=True, autoincrement=True)
    annotation_type = Column(Enum(ItsperAnnotationTypes), nullable=False)

    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    inference_image_id = Column(Integer, ForeignKey("inference_images.id"), nullable=True)  # Optional

    image = relationship("Image", back_populates="annotations")
    inference_image = relationship("InferenceImage", back_populates="annotations")


class Patient(Base):
    """Patient table."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    created = Column(DateTime(timezone=True), default=func.now())
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    patient_code = Column(String, unique=True)
    manifest_id = Column(Integer, ForeignKey("manifest.id"))

    manifest: Mapped["ItsperManifest"] = relationship("ItsperManifest", back_populates="patients")
    images: Mapped[List["Image"]] = relationship("Image", back_populates="patient")
    inference_images: Mapped[List["InferenceImage"]] = relationship("InferenceImage", back_populates="patient")


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    mpp = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    reader = Column(String, nullable=False)
    overwrite_mpp = Column(Float, nullable=True)

    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # Relationships
    annotations = relationship("Annotation", back_populates="image")
    inference_image = relationship("InferenceImage", back_populates="image", uselist=False)  # One-to-one relationship
    human_itsp = relationship("ITSPScore", back_populates="image")
    patient = relationship("Patient", back_populates="images")


class InferenceImage(Base):
    __tablename__ = "inference_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    mpp = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    reader = Column(String, nullable=False)
    tile_size = Column(Integer, nullable=False)

    # Foreign key linking to the Image table
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    # ForeignKey to link to Patient table
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    # Relationships
    annotations = relationship("Annotation", back_populates="inference_image")
    image = relationship("Image", back_populates="inference_image")  # Reference to the Image table
    patient = relationship("Patient", back_populates="inference_images")


class ITSPScore(Base):
    __tablename__ = "human_itsp"

    id = Column(Integer, primary_key=True, autoincrement=True)
    score = Column(Float, nullable=True)  # ITSP score given by a human observer
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)

    image = relationship("Image", back_populates="human_itsp")


def create_session(db_path: Path):
    """
    Creates the database for computing itsp database and its schema.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    return session()


def open_db_session(database_path: Path):
    """Open a session for interacting with the prediction database."""
    engine = create_engine(f"sqlite:///{database_path}")
    session = sessionmaker(bind=engine)
    return session()


def get_paired_data(session):
    """Retrieve paired tuples of image, inference image, annotation, and ITSP score."""
    paired_data = []

    # Querying for all images in the database
    images = session.query(Image).all()

    for image in images:
        # Get corresponding inference image
        inference_image = session.query(InferenceImage).filter_by(image_id=image.id).first()
        # Get corresponding annotation (if you have an Annotation table in the schema)
        annotation = session.query(Annotation).filter_by(image_id=image.id).first()
        # Get ITSP score if available
        itsp_score = session.query(ITSPScore).filter_by(image_id=image.id).first()

        # Create a tuple of the retrieved data (image, inference image, annotation, ITSP score)
        paired_data.append((image, inference_image, annotation, itsp_score))

    return paired_data


def summarize_database(session):
    """Print a summary of the database contents using rich formatting and logger."""
    logger.info("Summarizing the database contents from the manifest...")
    manifest_name = session.query(ItsperManifest).first().name
    total_images = session.query(Image).count()
    total_inference_images = session.query(InferenceImage).count()
    total_annotations = session.query(Annotation).count()
    total_itsp_scores = session.query(ITSPScore).count()
    annotation_type = session.query(Annotation.annotation_type).distinct().all()[0][0].value
    # Create a rich table for formatted output
    table = Table(title="Database Summary")

    # Define table columns
    table.add_column("Category", style="bold magenta")
    table.add_column("Count", justify="right")

    # Add rows with the summary information
    table.add_row("Manifest Name", manifest_name)
    table.add_row("Total Images", str(total_images))
    table.add_row("Total Inference Images", str(total_inference_images))
    table.add_row("Total Annotations", str(total_annotations))
    table.add_row("Type of Annotations", annotation_type)
    table.add_row("Total ITSP Scores", str(total_itsp_scores))
    console.print(table)
