/**
 * Supabase Frontend Client Initialization
 * 이 파일은 브라우저에서 Supabase와 직접 통신하기 위해 사용됩니다.
 */

// Supabase 접속 정보
const SUPABASE_URL = "https://xeyukdgbvvgfolcoluxo.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhleXVrZGdidnZnZm9sY29sdXhvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg5MDY2MzQsImV4cCI6MjA4NDQ4MjYzNH0.JHI5OVlUbxHXPhYfDCydME6aeo5c2S24wak88O40yuc";

// CDN을 통해 로드된 supabase 클라이언트 초기화
let supabaseClient = null;

if (typeof supabase !== 'undefined') {
    supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_KEY);
    console.log("✅ Supabase Frontend Client Initialized");
} else {
    console.error("❌ Supabase library not found. CDN script를 확인하세요.");
}

// 데이터를 저장하는 예시 함수
async function saveQSortResult(data) {
    const { respondent_name, sort_data, duration, interview_responses } = data;

    const { data: result, error } = await supabaseClient
        .from('q_sort_results')
        .insert([
            {
                respondent_name,
                sort_data,
                duration,
                interview_responses
            }
        ]);

    if (error) {
        console.error("❌ 데이터 저장 실패:", error);
        return null;
    }

    console.log("✅ 데이터 저장 성공:", result);
    return result;
}
