from sqlalchemy import create_engine, Text, Column, DateTime, Enum, ForeignKey, Integer, String, func, UniqueConstraint, Float
from sqlalchemy.orm import Mapped, relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pathlib import Path
from typing import List
from itsper.types import ItsperAnnotationTypes


Base = declarative_base()


class ItsperManifest(Base):
    __tablename__ = "manifest"

    id = Column(Integer, primary_key=True)
    created = Column(DateTime(timezone=True), default=func.now())
    last_updated = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    name = Column(String, unique=True)

    patients = relationship("Patient", back_populates="manifest")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    annotation_type = Column(Enum(ItsperAnnotationTypes), nullable=False)
    data = Column(Text, nullable=False)  # This could store JSON-like data for x, y coordinates or other properties

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
    score = Column(Float, nullable=False)  # ITSP score given by a human observer
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
