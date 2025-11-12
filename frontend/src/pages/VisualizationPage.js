import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup, Polyline, Circle } from 'react-leaflet';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { ArrowLeft, Brain, Loader2, MapPin, Activity, Clock, Radio, User } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { toast } from 'sonner';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Fix Leaflet default marker icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b'];

const VisualizationPage = () => {
  const { uploadId } = useParams();
  const navigate = useNavigate();
  const [records, setRecords] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState(null);

  useEffect(() => {
    fetchData();
  }, [uploadId]);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch records
      const recordsResponse = await axios.get(`${API}/records/${uploadId}`);
      if (recordsResponse.data.success) {
        setRecords(recordsResponse.data.records);
      }

      // Fetch statistics
      const statsResponse = await axios.get(`${API}/statistics/${uploadId}`);
      if (statsResponse.data.success) {
        setStatistics(statsResponse.data.statistics);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
      toast.error('Gagal memuat data');
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    try {
      const response = await axios.post(`${API}/analyze`, {
        upload_id: uploadId,
        analysis_type: 'comprehensive'
      });

      if (response.data.success) {
        setAnalysis(response.data.analysis);
        toast.success(`Analisis selesai! ${response.data.interpolated_count} GPS points diinterpolasi`);
        // Refresh data to get interpolated records
        fetchData();
      }
    } catch (error) {
      console.error('Error analyzing:', error);
      toast.error('Gagal melakukan analisis: ' + (error.response?.data?.detail || error.message));
    } finally {
      setAnalyzing(false);
    }
  };

  // Filter records with valid GPS
  const validGpsRecords = useMemo(() => {
    return records.filter(r => r.a_lat && r.a_long);
  }, [records]);

  // Calculate map center and bounds
  const mapCenter = useMemo(() => {
    if (statistics?.gps_bounds) {
      return [statistics.gps_bounds.center_lat, statistics.gps_bounds.center_long];
    }
    return [-6.9175, 107.6191]; // Default: Bandung
  }, [statistics]);

  // Prepare chart data
  const activityChartData = useMemo(() => {
    if (!statistics?.activity_distribution) return [];
    return Object.entries(statistics.activity_distribution).map(([name, value]) => ({
      name,
      value
    }));
  }, [statistics]);

  const directionChartData = useMemo(() => {
    if (!statistics?.direction_distribution) return [];
    return Object.entries(statistics.direction_distribution).map(([name, value]) => ({
      name,
      value
    }));
  }, [statistics]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #e0f2f7 0%, #f5f7fa 50%, #fce4ec 100%)' }}>
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4" style={{ color: '#667eea' }} />
          <p className="text-lg font-semibold">Memuat data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(135deg, #e0f2f7 0%, #f5f7fa 50%, #fce4ec 100%)' }}>
      {/* Header */}
      <div className="glass-card border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                data-testid="back-button"
                variant="outline"
                onClick={() => navigate('/')}
                className="flex items-center space-x-2"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Kembali</span>
              </Button>
              <div>
                <h1 className="text-2xl font-bold gradient-text">Visualisasi Trajectory</h1>
                <p className="text-sm text-gray-600">{records.length} records</p>
              </div>
            </div>
            <Button
              data-testid="analyze-button"
              onClick={handleAnalyze}
              disabled={analyzing}
              className="flex items-center space-x-2"
              style={{
                background: analyzing ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none'
              }}
            >
              {analyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Menganalisis...</span>
                </>
              ) : (
                <>
                  <Brain className="w-4 h-4" />
                  <span>Analisis dengan AI</span>
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <Tabs defaultValue="map" className="space-y-6">
          <TabsList className="glass-card">
            <TabsTrigger value="map" data-testid="map-tab">Peta Trajectory</TabsTrigger>
            <TabsTrigger value="statistics" data-testid="statistics-tab">Statistik</TabsTrigger>
            <TabsTrigger value="analysis" data-testid="analysis-tab">Analisis AI</TabsTrigger>
            <TabsTrigger value="data" data-testid="data-tab">Data Detail</TabsTrigger>
          </TabsList>

          {/* Map Tab */}
          <TabsContent value="map" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <MapPin className="w-5 h-5" style={{ color: '#667eea' }} />
                  <span>Peta Trajectory GPS</span>
                </CardTitle>
                <CardDescription>
                  Visualisasi pergerakan pengguna berdasarkan koordinat GPS dari CDR
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div style={{ height: '600px', borderRadius: '12px', overflow: 'hidden' }} data-testid="map-container">
                  <MapContainer
                    center={mapCenter}
                    zoom={13}
                    style={{ height: '100%', width: '100%' }}
                  >
                    <TileLayer
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    
                    {/* Trajectory line */}
                    {validGpsRecords.length > 1 && (
                      <Polyline
                        positions={validGpsRecords.map(r => [r.a_lat, r.a_long])}
                        color="#667eea"
                        weight={3}
                        opacity={0.7}
                      />
                    )}
                    
                    {/* Markers */}
                    {validGpsRecords.map((record, index) => (
                      <Marker
                        key={record.id || index}
                        position={[record.a_lat, record.a_long]}
                        eventHandlers={{
                          click: () => setSelectedRecord(record)
                        }}
                      >
                        <Popup>
                          <div className="space-y-2">
                            <div className="font-bold text-sm" style={{ color: '#667eea' }}>
                              Record #{index + 1}
                            </div>
                            <div className="text-xs space-y-1">
                              <div><strong>Waktu:</strong> {record.time || 'N/A'}</div>
                              <div><strong>Latitude:</strong> {record.a_lat?.toFixed(6)}</div>
                              <div><strong>Longitude:</strong> {record.a_long?.toFixed(6)}</div>
                              <div><strong>Cell Tower:</strong> {record.a_lac_cid || 'N/A'}</div>
                              <div><strong>User:</strong> {record.b_number || 'N/A'}</div>
                              <div><strong>Activity:</strong> {record.calltype || 'N/A'}</div>
                              {record.interpolated && (
                                <div className="text-orange-600"><strong>⚠ GPS Diinterpolasi</strong></div>
                              )}
                            </div>
                          </div>
                        </Popup>
                      </Marker>
                    ))}
                    
                    {/* Start and End markers with different colors */}
                    {validGpsRecords.length > 0 && (
                      <>
                        <Circle
                          center={[validGpsRecords[0].a_lat, validGpsRecords[0].a_long]}
                          radius={100}
                          pathOptions={{ color: '#43e97b', fillColor: '#43e97b', fillOpacity: 0.5 }}
                        />
                        <Circle
                          center={[validGpsRecords[validGpsRecords.length - 1].a_lat, validGpsRecords[validGpsRecords.length - 1].a_long]}
                          radius={100}
                          pathOptions={{ color: '#fa709a', fillColor: '#fa709a', fillOpacity: 0.5 }}
                        />
                      </>
                    )}
                  </MapContainer>
                </div>
              </CardContent>
            </Card>

            {/* Selected Record Details */}
            {selectedRecord && (
              <Card className="glass-card animate-fade-in">
                <CardHeader>
                  <CardTitle className="text-lg">Detail Record Terpilih</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 flex items-center space-x-1">
                        <Clock className="w-3 h-3" />
                        <span>Timestamp</span>
                      </div>
                      <div className="font-semibold">{selectedRecord.time || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 flex items-center space-x-1">
                        <MapPin className="w-3 h-3" />
                        <span>Latitude</span>
                      </div>
                      <div className="font-semibold">{selectedRecord.a_lat?.toFixed(6) || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 flex items-center space-x-1">
                        <MapPin className="w-3 h-3" />
                        <span>Longitude</span>
                      </div>
                      <div className="font-semibold">{selectedRecord.a_long?.toFixed(6) || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 flex items-center space-x-1">
                        <Radio className="w-3 h-3" />
                        <span>Cell Tower ID</span>
                      </div>
                      <div className="font-semibold text-xs">{selectedRecord.a_lac_cid || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 flex items-center space-x-1">
                        <User className="w-3 h-3" />
                        <span>User ID</span>
                      </div>
                      <div className="font-semibold text-xs">{selectedRecord.b_number || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500 flex items-center space-x-1">
                        <Activity className="w-3 h-3" />
                        <span>Activity Type</span>
                      </div>
                      <div className="font-semibold">{selectedRecord.calltype || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500">Direction</div>
                      <div className="font-semibold">{selectedRecord.direction || 'N/A'}</div>
                    </div>
                    <div className="space-y-1">
                      <div className="text-xs text-gray-500">Site Name</div>
                      <div className="font-semibold text-xs truncate">{selectedRecord.a_sitename || 'N/A'}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Statistics Tab */}
          <TabsContent value="statistics" className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-sm">Total Records</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold" style={{ color: '#667eea' }}>
                    {statistics?.total_records || 0}
                  </div>
                </CardContent>
              </Card>
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-sm">Records dengan GPS</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold" style={{ color: '#43e97b' }}>
                    {statistics?.records_with_gps || 0}
                  </div>
                </CardContent>
              </Card>
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-sm">GPS Hilang</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold" style={{ color: '#fa709a' }}>
                    {statistics?.missing_gps || 0}
                  </div>
                </CardContent>
              </Card>
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="text-sm">% GPS Hilang</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold" style={{ color: '#f093fb' }}>
                    {statistics?.missing_gps_percentage || 0}%
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Activity Distribution */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Distribusi Tipe Aktivitas</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={activityChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#667eea" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Direction Distribution */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle>Distribusi Arah Komunikasi</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={directionChartData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {directionChartData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            {!analysis ? (
              <Card className="glass-card">
                <CardContent className="py-16">
                  <div className="text-center space-y-4">
                    <Brain className="w-16 h-16 mx-auto" style={{ color: '#667eea' }} />
                    <h3 className="text-xl font-bold">Belum Ada Analisis AI</h3>
                    <p className="text-gray-600">Klik tombol "Analisis dengan AI" untuk memulai analisis trajectory dengan GPT-5</p>
                    <Button
                      data-testid="analyze-button-tab"
                      onClick={handleAnalyze}
                      disabled={analyzing}
                      className="mt-4"
                      style={{
                        background: analyzing ? '#9ca3af' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        color: 'white',
                        border: 'none'
                      }}
                    >
                      {analyzing ? 'Menganalisis...' : 'Mulai Analisis'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-6">
                <Card className="glass-card">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Brain className="w-5 h-5" style={{ color: '#667eea' }} />
                      <span>Hasil Analisis AI (GPT-5)</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      {Object.entries(analysis).map(([key, value]) => (
                        <div key={key} className="space-y-2">
                          <h4 className="font-semibold text-lg capitalize" style={{ color: '#667eea' }}>
                            {key.replace(/_/g, ' ')}
                          </h4>
                          <div className="p-4 rounded-lg" style={{ background: 'rgba(102, 126, 234, 0.05)' }}>
                            {typeof value === 'object' ? (
                              <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(value, null, 2)}</pre>
                            ) : (
                              <p className="text-sm text-gray-700">{value}</p>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          {/* Data Tab */}
          <TabsContent value="data" className="space-y-6">
            <Card className="glass-card">
              <CardHeader>
                <CardTitle>Data CDR Detail</CardTitle>
                <CardDescription>Semua records dari file yang diupload</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b" style={{ borderColor: '#667eea' }}>
                        <th className="p-2 text-left">#</th>
                        <th className="p-2 text-left">Time</th>
                        <th className="p-2 text-left">Lat</th>
                        <th className="p-2 text-left">Long</th>
                        <th className="p-2 text-left">Cell Tower</th>
                        <th className="p-2 text-left">User</th>
                        <th className="p-2 text-left">Activity</th>
                        <th className="p-2 text-left">Direction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {records.slice(0, 50).map((record, index) => (
                        <tr key={record.id || index} className="border-b hover:bg-gray-50">
                          <td className="p-2">{index + 1}</td>
                          <td className="p-2">{record.time || 'N/A'}</td>
                          <td className="p-2">{record.a_lat?.toFixed(6) || 'N/A'}</td>
                          <td className="p-2">{record.a_long?.toFixed(6) || 'N/A'}</td>
                          <td className="p-2 text-xs">{record.a_lac_cid || 'N/A'}</td>
                          <td className="p-2 text-xs">{record.b_number || 'N/A'}</td>
                          <td className="p-2">{record.calltype || 'N/A'}</td>
                          <td className="p-2">{record.direction || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {records.length > 50 && (
                    <div className="mt-4 text-center text-sm text-gray-600">
                      Menampilkan 50 dari {records.length} records
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default VisualizationPage;
