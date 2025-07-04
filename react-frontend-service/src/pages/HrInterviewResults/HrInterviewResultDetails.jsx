import React, {useContext, useEffect, useState} from "react";
import {useNavigate, useParams} from "react-router-dom";
import "./HrInterviewResults.css";
import {fetchWithAuth} from "../../api/fetchWithAuth";
import {AuthContext} from "../AuthProvider";

export const HrInterviewResultDetail = () => {
    const {attemptId} = useParams();
    const navigate = useNavigate();
    const [resultData, setResultData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState("");
    const {token} = useContext(AuthContext);
    const [feedbackObject, setFeedbackObject] = useState({});
    // Состояния для модального окна
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalImageVideo, setModalImageVideo] = useState('');
    const [modalImageAudio, setModalImageAudio] = useState('');
    const [showFullTranscript, setShowFullTranscript] = useState(false);
    const [showVideoEmotions, setShowVideoEmotions] = useState(false);
    const [showAudioEmotions, setShowAudioEmotions] = useState(false);

    useEffect(() => {
        fetchResultDetails();
    }, [attemptId, token]);

    const fetchResultDetails = async () => {
        try {
            setLoading(true);
            const response = await fetchWithAuth(`/hr-interview/results/${attemptId}`,
                {method: "GET"},
                token
            );

            if (response.ok) {
                const data = await response.json();
                setResultData(data);

                console.log("--- Инициализация: Логирование объектов answer.feedback ---");
                try {
                    // Предполагаем, что data.feedback - это строка, содержащая JSON-строку
                    const parsedFeedbackString = JSON.parse(data.feedback);
                    console.log(`Разобранный feedback:`, parsedFeedbackString);
                    setFeedbackObject(parsedFeedbackString);
                } catch (e) {
                    console.error(` Ошибка парсинга JSON для feedback:`, e);
                    // Если парсинг не удался, можно попробовать установить feedbackObject как пустой объект
                    // или как результат одного JSON.parse, если data.feedback - это просто JSON-строка
                    try {
                        const singleParsedFeedback = JSON.parse(data.feedback);
                        setFeedbackObject(singleParsedFeedback);
                        console.log(`Разобранный feedback (один парсинг):`, singleParsedFeedback);
                    } catch (singleParseError) {
                        console.error(` Ошибка парсинга JSON для feedback (один парсинг):`, singleParseError);
                        setFeedbackObject({}); // Установка пустого объекта в случае ошибки
                    }
                }
            } else {
                const errorData = await response.json();
                setErrorMessage(`Ошибка при загрузке деталей: ${errorData.error || 'Неизвестная ошибка'}`);
            }
        } catch (error) {
            setErrorMessage(`Произошла ошибка: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    const handleBack = () => {
        navigate('/hr-interview/results');
    };

    // Функция для форматирования даты в читаемом виде
    const formatDate = (dateString) => {
        if (!dateString) return "Нет данных";
        const date = new Date(dateString);
        return date.toLocaleString();
    };

    // Функция для определения класса статуса
    const getStatusClass = (status) => {
        if (!status) return '';
        status = status.toLowerCase();
        if (status === 'completed' || status === 'завершено') return 'status-completed';
        if (status === 'pending' || status === 'в ожидании') return 'status-pending';
        if (status === 'failed' || status === 'неудачно') return 'status-failed';
        return '';
    };

    // Функции для управления модальным окном
    const openModal = (imageBytesVideo, imageBytesAudio) => {
        setModalImageVideo(`data:image/jpeg;base64,${imageBytesVideo}`);
        setModalImageAudio(`data:image/jpeg;base64,${imageBytesAudio}`);
        setIsModalOpen(true);
    };


    const closeModal = () => {
        setIsModalOpen(false);
        setModalImageVideo('');
        setModalImageAudio('');
    };

    if (loading) {
        return <div className="vacancy-container"><h1>Загрузка результатов собеседования...</h1></div>;
    }

    return (
        <div className="vacancy-container">
            <button onClick={handleBack} className="back-button">← Назад к списку</button>
            <h1 className="page-title">Результаты собеседования #{attemptId}</h1>

            {errorMessage && <div className="error-message">{errorMessage}</div>}

            {resultData && (
                <>
                    <div className="interview-result-detail">
                        <div className="result-summary">
                            <h2>Общая информация</h2>
                            <table>
                                <tbody>
                                <tr>
                                    <th>Дата прохождения:</th>
                                    <td>{formatDate(resultData.startTime)}</td>
                                </tr>
                                <tr>
                                    <th>Статус:</th>
                                    <td className={getStatusClass(resultData.status)}>
                                        {resultData.status.toLowerCase() === 'completed' ? 'Завершено' : resultData.status.toLowerCase() === 'pending' ? 'В ожидании' : 'Неудачно'}
                                    </td>
                                </tr>
                                    </tbody>
                            </table>
                        </div>
                    </div>


                    {/* Раздел для кнопки открытия графика */}
                    {resultData.videoImageData && (
                        <div className="interview-result-detail plot-section">
                            <h2>График</h2>
                            <button onClick={() => openModal(resultData.videoImageData, resultData.audioImageData)}
                                    className="button-show-plot">
                                Показать график
                            </button>
                        </div>
                    )}

                    <div className="interview-result-detail">
                        <div className="result-summary">
                            <h2>Статистика</h2>
                            <table>
                                <tbody>
                                <tr>
                                    <th>Длительность</th>
                                    <td>{parseInt(feedbackObject.stats.video_duration_sec)} c.</td>
                                </tr>
                                <tr>
                                    <th>Темп речи</th>
                                    <td>{parseInt(feedbackObject.stats.avg_wpm)} cл/мин</td>
                                </tr>
                                <tr>
                                    <th>Общее время пауз</th>
                                    <td>{parseInt(feedbackObject.stats.total_pause_time_sec)} с.</td>
                                </tr>
                                <tr>
                                    <th>Частота морганий</th>
                                    <td>{parseInt(feedbackObject.stats.blink_rate_per_minute)} морганий/сек</td>
                                </tr>
                                <tr>
                                    <th>Число символов в речи</th>
                                    <td>{parseInt(feedbackObject.stats.transcript_length_chars)}</td>
                                </tr>
                                <tr>
                                    <th>Расшифровка речи:</th>
                                    <td>
                                        <button
                                            onClick={() => setShowFullTranscript(!showFullTranscript)}
                                            className="button-show-transcript" // Можете добавить стили для этой кнопки
                                        >
                                            {showFullTranscript ? "Скрыть" : "Показать"}
                                        </button>
                                        {showFullTranscript && (
                                            <div style={{ marginTop: '10px', whiteSpace: 'pre-wrap', textAlign: 'left', maxHeight: '300px', overflowY: 'auto', border: '1px solid #eee', padding: '10px' }}>
                                                {feedbackObject.stats.transcript_snippet}
                                            </div>
                                        )}
                                    </td>
                                </tr>
                                <tr>
                                    <th>Общее время улыбок</th>
                                    <td>{parseInt(feedbackObject.stats.total_smile_duration_seconds)} с.</td>
                                </tr>
                                <tr>
                                    <th>Доля времени с улыбкой</th>
                                    <td>{parseInt(feedbackObject.stats.smile_percentage_of_face_time)}%</td>
                                </tr>
                                <tr>
                                    <th>Token-Type Ratio - лексическое разнообразие</th>
                                    <td>{feedbackObject.stats.ttr_lexical_diversity.toFixed(2)}</td>
                                </tr>
                                <tr>
                                    <th>Доля отведенного взгляда</th>
                                    <td>{parseInt(feedbackObject.stats.gaze_aversion_percentage_of_face_time)}%</td>
                                </tr>
                                <tr>
                                    <th>Распределение эмоций (видео):</th>
                                    <td>
                                        <button
                                            onClick={() => setShowVideoEmotions(!showVideoEmotions)}
                                            className="button-show-transcript" // Можете добавить стили для этой кнопки
                                        >
                                            {showVideoEmotions ? "Скрыть" : "Показать"}
                                        </button>
                                        {showVideoEmotions && (
                                            <div style={{ marginTop: '10px', whiteSpace: 'pre-wrap', textAlign: 'left', maxHeight: '300px', overflowY: 'auto', border: '1px solid #eee', padding: '10px' }}>
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <th>Злость</th>
                                                        <td>{parseInt(feedbackObject.stats.emotion_distribution.angry)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Страх</th>
                                                        <td>{parseInt(feedbackObject.stats.emotion_distribution.fear)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Грусть</th>
                                                        <td>{parseInt(feedbackObject.stats.emotion_distribution.sad)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Нейтральная</th>
                                                        <td>{parseInt(feedbackObject.stats.emotion_distribution.neutral)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Удивление</th>
                                                        <td>{parseInt(feedbackObject.stats.emotion_distribution.surprise)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Позитив</th>
                                                        <td>{parseInt(feedbackObject.stats.emotion_distribution.happy)}%</td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                    </td>
                                </tr>
                                <tr>
                                    <th>Распределение эмоций (аудио):</th>
                                    <td>
                                        <button
                                            onClick={() => setShowAudioEmotions(!showAudioEmotions)}
                                            className="button-show-transcript" // Можете добавить стили для этой кнопки
                                        >
                                            {showAudioEmotions ? "Скрыть" : "Показать"}
                                        </button>
                                        {showAudioEmotions && (
                                            <div style={{ marginTop: '10px', whiteSpace: 'pre-wrap', textAlign: 'left', maxHeight: '500px', overflowY: 'auto', border: '1px solid #eee', padding: '10px' }}>
                                                <table>
                                                    <tbody>
                                                    <tr>
                                                        <th>Злость</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.anger * 100)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Отвращение</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.disgust * 100)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Страх</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.fear * 100)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Грусть</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.sadness * 100)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Позитив</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.neutral * 100)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Счастье</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.happiness * 100)}%</td>
                                                    </tr>
                                                    <tr>
                                                        <th>Энтузиазм</th>
                                                        <td>{parseInt(feedbackObject.audio_stats.average_probabilities.enthusiasm * 100)}%</td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                    </td>
                                </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {feedbackObject.advice && (
                        <div className="interview-result-detail recommendations">
                            <h2>Рекомендации от AI</h2>
                            <p>{feedbackObject.advice}</p>
                        </div>
                    )}
                </>
            )}

            {isModalOpen && (
                <div className="modal-overlay" onClick={closeModal}>
                    <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                        <span className="modal-close" onClick={closeModal}>&times;</span>
                        <img src={modalImageVideo} alt="График" style={{maxWidth: '100%', maxHeight: '80vh'}}/>
                        <span className="modal-close" onClick={closeModal}>&times;</span>
                        <img src={modalImageAudio} alt="График" style={{maxWidth: '100%', maxHeight: '80vh'}}/>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HrInterviewResultDetail;