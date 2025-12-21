from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class MedicalFlashcard(Base):
    """AYA NAIM: Store medical flashcards in SQLite"""
    __tablename__ = 'medical_flashcards'
    
    id = Column(Integer, primary_key=True)
    instruction = Column(Text)
    input_text = Column(Text)
    output = Column(Text)
    instruction_clean = Column(Text)
    output_clean = Column(Text)
    needs_summary = Column(Integer)  # 0 or 1
    created_at = Column(DateTime, default=datetime.now)
    metadata_json = Column(Text)  # Store additional info

class MemoryStore:
    """AYA NAIM: Database operations for medical data"""
    
    def __init__(self, db_path='medical_flashcards.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        print(f"AYA: Memory store connected to {db_path}")
    
    def save_flashcard(self, instruction, input_text, output, instruction_clean, output_clean, needs_summary):
        """Save a medical flashcard to database"""
        card = MedicalFlashcard(
            instruction=instruction,
            input_text=input_text,
            output=output,
            instruction_clean=instruction_clean,
            output_clean=output_clean,
            needs_summary=1 if needs_summary else 0,
            metadata_json=json.dumps({'source': 'medical_meadow'})
        )
        self.session.add(card)
        self.session.commit()
        return card.id
    
    def get_all_flashcards(self):
        """Get all flashcards from database"""
        return self.session.query(MedicalFlashcard).all()
    
    def get_stats(self):
        """Get database statistics"""
        total = self.session.query(MedicalFlashcard).count()
        need_summary = self.session.query(MedicalFlashcard).filter(MedicalFlashcard.needs_summary == 1).count()
        
        return {
            'total_flashcards': total,
            'need_summary': need_summary,
            'summary_percentage': (need_summary / total * 100) if total > 0 else 0
        }