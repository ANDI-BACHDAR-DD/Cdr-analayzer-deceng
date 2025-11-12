import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileSpreadsheet, TrendingUp, MapPin, Activity } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const HomePage = () => {
  const [uploads, setUploads] = useState([]);
  const [uploading, setUploading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetchUploads();
  }, []);

  const fetchUploads = async () => {
    try {
      const response = await axios.get(`${API}/uploads`);
      if (response.data.success) {
        setUploads(response.data.uploads);
      }
    } catch (error) {
      console.error('Error fetching uploads:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.xls')) {
      toast.error('Hanya file XLSX/XLS yang didukung');
      return;
    }

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (response.data.success) {
        toast.success(`File berhasil diupload! ${response.data.total_records} records ditemukan`);
        fetchUploads();
        
        // Navigate to visualization page
        setTimeout(() => {
          navigate(`/visualization/${response.data.upload_id}`);
        }, 1000);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error('Gagal mengupload file: ' + (error.response?.data?.detail || error.message));
    } finally {
      setUploading(false);
    }
  };

  const handleViewVisualization = (uploadId) => {
    navigate(`/visualization/${uploadId}`);
  };

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(135deg, #e0f2f7 0%, #f5f7fa 50%, #fce4ec 100%)' }}>
      {/* Header */}
      <div className="glass-card border-b sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center space-x-4">
            <div className="p-3 rounded-2xl" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              <Activity className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold gradient-text">CDR Trajectory Analyzer</h1>
              <p className="text-sm text-gray-600 mt-1">Analisis Trajectory Manusia Berbasis AI dari Data CDR Telekomunikasi</p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Upload Section */}
        <div className="animate-fade-in">
          <Card className="glass-card border-2 border-dashed" style={{ borderColor: '#667eea' }}>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Upload className="w-6 h-6" style={{ color: '#667eea' }} />
                <span>Upload Data CDR</span>
              </CardTitle>
              <CardDescription>
                Upload file XLSX berisi data Call Detail Record (CDR) dengan informasi GPS, timestamp, dan aktivitas pengguna
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col items-center justify-center py-12 space-y-4">
                <div className="p-6 rounded-full" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
                  <FileSpreadsheet className="w-12 h-12 text-white" />
                </div>
                <div className="text-center">
                  <h3 className="text-lg font-semibold mb-2">Upload File CDR</h3>
                  <p className="text-sm text-gray-600 mb-4">Format: XLSX atau XLS</p>
                </div>
                <label htmlFor="file-upload">
                  <Button
                    data-testid="upload-cdr-button"
                    disabled={uploading}
                    className="cursor-pointer"
                    style={{
                      background: uploading ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                      color: 'white',
                      border: 'none'
                    }}
                    asChild
                  >
                    <span>
                      {uploading ? 'Mengupload...' : 'Pilih File'}
                    </span>
                  </Button>
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept=".xlsx,.xls"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={uploading}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Uploaded Files List */}
        {uploads.length > 0 && (
          <div className="mt-12 animate-slide-in">
            <h2 className="text-2xl font-bold mb-6 flex items-center space-x-2">
              <TrendingUp className="w-6 h-6" style={{ color: '#667eea' }} />
              <span>Data Terupload</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {uploads.map((upload, index) => (
                <Card
                  key={upload.upload_id}
                  className="glass-card hover:shadow-xl cursor-pointer"
                  style={{
                    animation: `fadeIn 0.5s ease-out ${index * 0.1}s both`,
                    transform: 'translateY(0)',
                    transition: 'all 0.3s ease'
                  }}
                  onClick={() => handleViewVisualization(upload.upload_id)}
                  data-testid={`upload-card-${index}`}
                >
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center space-x-2">
                      <MapPin className="w-5 h-5" style={{ color: '#667eea' }} />
                      <span className="truncate">{upload.filename}</span>
                    </CardTitle>
                    <CardDescription>
                      {new Date(upload.uploaded_at).toLocaleDateString('id-ID', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 rounded-lg" style={{ background: 'rgba(102, 126, 234, 0.1)' }}>
                        <span className="text-sm font-medium text-gray-700">Total Records</span>
                        <span className="text-lg font-bold" style={{ color: '#667eea' }}>{upload.total_records}</span>
                      </div>
                      <Button
                        data-testid={`view-visualization-${index}`}
                        className="w-full"
                        style={{
                          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                          color: 'white',
                          border: 'none'
                        }}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleViewVisualization(upload.upload_id);
                        }}
                      >
                        Lihat Visualisasi
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Features Section */}
        <div className="mt-16 animate-fade-in">
          <h2 className="text-2xl font-bold text-center mb-8">Fitur Utama</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                icon: MapPin,
                title: 'Visualisasi Peta',
                description: 'Tampilkan trajectory pergerakan di peta interaktif dengan Leaflet/OpenStreetMap'
              },
              {
                icon: TrendingUp,
                title: 'Analisis AI',
                description: 'Ekstraksi pola pergerakan dan prediksi trajectory menggunakan GPT-5'
              },
              {
                icon: Activity,
                title: 'Interpolasi Data',
                description: 'Isi otomatis data GPS yang hilang dengan algoritma interpolasi cerdas'
              }
            ].map((feature, index) => (
              <Card key={index} className="glass-card text-center">
                <CardHeader>
                  <div className="mx-auto p-4 rounded-2xl w-fit" style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>
                  <CardTitle className="mt-4">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
