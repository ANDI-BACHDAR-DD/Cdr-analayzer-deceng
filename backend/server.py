from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import openpyxl
import pandas as pd
import io
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json
import numpy as np
from sklearn.linear_model import LinearRegression

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Models
class CDRRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    upload_id: str
    date: Optional[str] = None
    time: Optional[str] = None
    duration: Optional[str] = None
    a_number: Optional[str] = None
    a_imei: Optional[str] = None
    a_imei_type: Optional[str] = None
    a_imsi: Optional[str] = None
    a_lac_cid: Optional[str] = None
    a_sitename: Optional[str] = None
    b_number: Optional[str] = None
    b_imei: Optional[str] = None
    b_imei_type: Optional[str] = None
    b_imsi: Optional[str] = None
    b_lac_cid: Optional[str] = None
    b_sitename: Optional[str] = None
    calltype: Optional[str] = None
    direction: Optional[str] = None
    c_number: Optional[str] = None
    a_lat: Optional[float] = None
    a_long: Optional[float] = None
    b_lat: Optional[float] = None
    b_long: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UploadMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    upload_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    total_records: int
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None

class TrajectoryAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    upload_id: str
    analysis_type: str  # pattern, interpolation, prediction
    result: Dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisRequest(BaseModel):
    upload_id: str
    analysis_type: str = "comprehensive"  # comprehensive, pattern, interpolation, prediction

# Helper Functions
def parse_cdr_file(file_content: bytes, filename: str) -> List[Dict]:
    """Parse CDR XLSX file and extract data"""
    try:
        # Load workbook
        wb = openpyxl.load_workbook(io.BytesIO(file_content))
        ws = wb.active
        
        # Get data as list of lists
        data = []
        for row in ws.iter_rows(values_only=True):
            data.append(row)
        
        # Check if file has data
        if len(data) < 2:
            return []
        
        # Get headers from first row
        headers = [str(h).strip().lower().replace('% ', '').replace(' ', '_') if h else f'col_{i}' for i, h in enumerate(data[0])]
        
        # Parse data rows
        records = []
        for row in data[1:]:  # Skip header
            if not row or all(cell is None or str(cell).strip() == '' for cell in row):
                continue
                
            record = {}
            for i, value in enumerate(row):
                if i < len(headers):
                    header = headers[i]
                    # Clean and convert value
                    if value is not None and str(value).strip() != '':
                        # Try to convert lat/long to float
                        if 'lat' in header or 'long' in header:
                            try:
                                record[header] = float(value)
                            except:
                                record[header] = None
                        else:
                            record[header] = str(value).strip()
                    else:
                        record[header] = None
            
            records.append(record)
        
        return records
    except Exception as e:
        logger.error(f"Error parsing CDR file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

async def interpolate_missing_gps(records: List[Dict]) -> List[Dict]:
    """Interpolate missing GPS coordinates using linear interpolation"""
    try:
        # Filter records with valid GPS
        valid_gps_records = [r for r in records if r.get('a_lat') and r.get('a_long')]
        
        if len(valid_gps_records) < 2:
            return records
        
        # Create time-based index for interpolation
        for i, record in enumerate(records):
            record['index'] = i
        
        # Prepare data for interpolation
        known_indices = [r['index'] for r in valid_gps_records]
        known_lats = [r['a_lat'] for r in valid_gps_records]
        known_longs = [r['a_long'] for r in valid_gps_records]
        
        # Simple linear interpolation for missing values
        for i, record in enumerate(records):
            if not record.get('a_lat') or not record.get('a_long'):
                # Find nearest known points
                if i < known_indices[0]:
                    # Use first known point
                    record['a_lat'] = known_lats[0]
                    record['a_long'] = known_longs[0]
                    record['interpolated'] = True
                elif i > known_indices[-1]:
                    # Use last known point
                    record['a_lat'] = known_lats[-1]
                    record['a_long'] = known_longs[-1]
                    record['interpolated'] = True
                else:
                    # Interpolate between points
                    for j in range(len(known_indices) - 1):
                        if known_indices[j] < i < known_indices[j + 1]:
                            # Linear interpolation
                            ratio = (i - known_indices[j]) / (known_indices[j + 1] - known_indices[j])
                            record['a_lat'] = known_lats[j] + ratio * (known_lats[j + 1] - known_lats[j])
                            record['a_long'] = known_longs[j] + ratio * (known_longs[j + 1] - known_longs[j])
                            record['interpolated'] = True
                            break
        
        return records
    except Exception as e:
        logger.error(f"Error interpolating GPS: {str(e)}")
        return records

async def analyze_with_ai(records: List[Dict], analysis_type: str) -> Dict[str, Any]:
    """Analyze trajectory data using GPT-5"""
    try:
        # Prepare summary data for AI analysis
        total_records = len(records)
        records_with_gps = len([r for r in records if r.get('a_lat') and r.get('a_long')])
        missing_gps = total_records - records_with_gps
        
        # Get unique locations
        unique_locations = set()
        for r in records:
            if r.get('a_lat') and r.get('a_long'):
                unique_locations.add((round(r['a_lat'], 4), round(r['a_long'], 4)))
        
        # Get activity types
        activity_types = {}
        for r in records:
            activity = r.get('calltype', 'UNKNOWN')
            activity_types[activity] = activity_types.get(activity, 0) + 1
        
        # Get time range
        times = [r.get('time', '') for r in records if r.get('time')]
        time_range = f"{min(times)} to {max(times)}" if times else "Unknown"
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze this CDR trajectory data:

Total Records: {total_records}
Records with GPS: {records_with_gps}
Missing GPS: {missing_gps}
Unique Locations: {len(unique_locations)}
Activity Types: {json.dumps(activity_types)}
Time Range: {time_range}

Sample Records (first 5):
{json.dumps(records[:5], indent=2)}

Provide a comprehensive analysis including:
1. Movement patterns identified
2. Key locations and hotspots
3. Activity patterns (temporal and spatial)
4. Data quality assessment
5. Privacy recommendations
6. Insights for trajectory prediction

Respond in JSON format with keys: movement_patterns, key_locations, activity_analysis, data_quality, privacy_notes, prediction_insights"""
        
        # Initialize LLM Chat with GPT-5
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"cdr_analysis_{uuid.uuid4()}",
            system_message="You are an expert in CDR trajectory analysis, human mobility patterns, and geospatial data science. Provide detailed, actionable insights."
        ).with_model("openai", "gpt-5")
        
        # Send message and get response
        user_message = UserMessage(text=analysis_prompt)
        response = await chat.send_message(user_message)
        
        # Parse AI response
        try:
            ai_analysis = json.loads(response)
        except:
            # If response is not JSON, structure it
            ai_analysis = {
                "raw_analysis": response,
                "total_records": total_records,
                "records_with_gps": records_with_gps,
                "missing_gps_percentage": round((missing_gps / total_records) * 100, 2) if total_records > 0 else 0,
                "unique_locations": len(unique_locations),
                "activity_distribution": activity_types
            }
        
        return ai_analysis
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return {
            "error": str(e),
            "message": "AI analysis failed, returning basic statistics",
            "total_records": len(records),
            "records_with_gps": len([r for r in records if r.get('a_lat') and r.get('a_long')])
        }

# API Routes
@api_router.get("/")
async def root():
    return {"message": "CDR Trajectory Analysis API", "version": "1.0.0"}

@api_router.post("/upload")
async def upload_cdr_file(file: UploadFile = File(...)):
    """Upload and parse CDR XLSX file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only XLSX/XLS files are supported")
        
        # Read file content
        file_content = await file.read()
        
        # Parse CDR file
        records = parse_cdr_file(file_content, file.filename)
        
        if not records:
            raise HTTPException(status_code=400, detail="No valid data found in file")
        
        # Create upload metadata
        upload_id = str(uuid.uuid4())
        metadata = UploadMetadata(
            upload_id=upload_id,
            filename=file.filename,
            total_records=len(records)
        )
        
        # Save metadata to database
        metadata_dict = metadata.model_dump()
        metadata_dict['uploaded_at'] = metadata_dict['uploaded_at'].isoformat()
        await db.upload_metadata.insert_one(metadata_dict)
        
        # Save CDR records to database
        for record in records:
            cdr_record = CDRRecord(upload_id=upload_id, **record)
            cdr_dict = cdr_record.model_dump()
            cdr_dict['timestamp'] = cdr_dict['timestamp'].isoformat()
            await db.cdr_records.insert_one(cdr_dict)
        
        logger.info(f"Uploaded {len(records)} CDR records with upload_id: {upload_id}")
        
        return {
            "success": True,
            "upload_id": upload_id,
            "filename": file.filename,
            "total_records": len(records),
            "message": "File uploaded and parsed successfully"
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/uploads")
async def get_uploads():
    """Get all uploaded files metadata"""
    try:
        uploads = await db.upload_metadata.find({}, {"_id": 0}).to_list(1000)
        return {"success": True, "uploads": uploads}
    except Exception as e:
        logger.error(f"Error getting uploads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/records/{upload_id}")
async def get_records(upload_id: str):
    """Get CDR records for a specific upload"""
    try:
        records = await db.cdr_records.find({"upload_id": upload_id}, {"_id": 0}).to_list(10000)
        return {"success": True, "upload_id": upload_id, "records": records, "count": len(records)}
    except Exception as e:
        logger.error(f"Error getting records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze")
async def analyze_trajectory(request: AnalysisRequest):
    """Analyze trajectory data with AI"""
    try:
        # Get records from database
        records = await db.cdr_records.find({"upload_id": request.upload_id}, {"_id": 0}).to_list(10000)
        
        if not records:
            raise HTTPException(status_code=404, detail="No records found for this upload_id")
        
        # Interpolate missing GPS coordinates
        logger.info(f"Interpolating missing GPS for {len(records)} records")
        interpolated_records = await interpolate_missing_gps(records)
        
        # Perform AI analysis
        logger.info(f"Performing AI analysis on {len(interpolated_records)} records")
        analysis_result = await analyze_with_ai(interpolated_records, request.analysis_type)
        
        # Save analysis to database
        analysis = TrajectoryAnalysis(
            upload_id=request.upload_id,
            analysis_type=request.analysis_type,
            result=analysis_result
        )
        
        analysis_dict = analysis.model_dump()
        analysis_dict['created_at'] = analysis_dict['created_at'].isoformat()
        await db.trajectory_analysis.insert_one(analysis_dict)
        
        # Update records with interpolated data
        for record in interpolated_records:
            if record.get('interpolated'):
                await db.cdr_records.update_one(
                    {"upload_id": request.upload_id, "id": record.get('id')},
                    {"$set": {"a_lat": record['a_lat'], "a_long": record['a_long'], "interpolated": True}}
                )
        
        return {
            "success": True,
            "upload_id": request.upload_id,
            "analysis": analysis_result,
            "interpolated_count": len([r for r in interpolated_records if r.get('interpolated')]),
            "total_records": len(interpolated_records)
        }
    except Exception as e:
        logger.error(f"Error analyzing trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/statistics/{upload_id}")
async def get_statistics(upload_id: str):
    """Get basic statistics for uploaded CDR data"""
    try:
        records = await db.cdr_records.find({"upload_id": upload_id}, {"_id": 0}).to_list(10000)
        
        if not records:
            raise HTTPException(status_code=404, detail="No records found")
        
        # Calculate statistics
        total_records = len(records)
        records_with_gps = len([r for r in records if r.get('a_lat') and r.get('a_long')])
        missing_gps = total_records - records_with_gps
        
        # Activity distribution
        activity_dist = {}
        for r in records:
            activity = r.get('calltype', 'UNKNOWN')
            activity_dist[activity] = activity_dist.get(activity, 0) + 1
        
        # Direction distribution
        direction_dist = {}
        for r in records:
            direction = r.get('direction', 'UNKNOWN')
            direction_dist[direction] = direction_dist.get(direction, 0) + 1
        
        # GPS bounds
        lats = [r['a_lat'] for r in records if r.get('a_lat')]
        longs = [r['a_long'] for r in records if r.get('a_long')]
        
        gps_bounds = None
        if lats and longs:
            gps_bounds = {
                "min_lat": min(lats),
                "max_lat": max(lats),
                "min_long": min(longs),
                "max_long": max(longs),
                "center_lat": sum(lats) / len(lats),
                "center_long": sum(longs) / len(longs)
            }
        
        return {
            "success": True,
            "upload_id": upload_id,
            "statistics": {
                "total_records": total_records,
                "records_with_gps": records_with_gps,
                "missing_gps": missing_gps,
                "missing_gps_percentage": round((missing_gps / total_records) * 100, 2),
                "activity_distribution": activity_dist,
                "direction_distribution": direction_dist,
                "gps_bounds": gps_bounds
            }
        }
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
