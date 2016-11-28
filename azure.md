# Azure Machine Learningを使った実験

## <a name="workspace">ワークスペースの作成</a>

1. [https://account.windowsazure.com](https://account.windowsazure.com)からログイン
1. 「ポータル」をクリック（ダッシュボードに移動する）
1. 左のパネルから「その他のサービス」を選択し、「インテルジェンス＋分析」内の「「Machine Learningワークスペース」を選択
1.  画面が変わったら「追加」ボタンを押す
1. 「ワークスペース名」に適当な名前を入力（ここでは「waseda」とする）
1. その下の「リソースグループ」で「新規作成」を選択肢、テキストボックスに適当な名前を入力（ここでは「wasedaresource」する）
1. 下の方にスクロールして、「Webサービスプラン価格レベル」を選択し、右に表示される「DEVTEST 標準」を押し、下の「選択」ボタンを押す
1. 下フレーム内の「作成」を押す
1. ワークスペース一覧画面に移るが、しばらくまって「更新」を押すと新しいワークスペースが表示される

## <a name="regression">自動車価格データを使った回帰モデルの実験</a>

ここでの手順はマイクロソフトが公開している[公式チュートリアル](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-create-experiment)に多少手を加えたものです。公式チュートリアルも適宜参考にしてください。

### 新しいExperiment（実験）を作る

1. ワークスペース一覧画面から上で作ったワークスペース名をクリック
1. 右のパネルに表示される「Machine Learning Studioの起動」をクリック
1. 画面が変わったら「Sign In」ボタンを押す（ここでまたサインインし直すのがインターフェース的に統一感がなくてイケてない:( ）
1. （ここで「The Studio failed to load. Please refresh this page.」というメッセージが出て、何度リロードしてもうまくいかないことがあるが、一度ブラウザを終了して再起動するとうまく動く）
1. 左のパネルから「EXPERIMENTS」を選択し、左下の「NEW」ボタンを押す
1. サンプルがいいろと出てくるので、一番左上の「Blank Experiment」を選択

### 自動車データを使った回帰モデルの実験

1. 左のパネルから「Saved Datasets」→「Samples」→「Automobile price data (Raw)」を選び、右のパネルにドラッグ
1. 「Data Transformation」→「Manipulation」→「Select Columns in Dataset」を選び、右のパネルにドラッグ
1. 右のパネルで、「Select Columns in Dataset」の入力（箱の上の丸）と「Automobile price data (Raw)」の出力（箱の下の丸）をつなぐ
