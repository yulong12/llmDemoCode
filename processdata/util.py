import os
import psutil
from transformers import AutoTokenizer  # 使用 Hugging Face 的 tokenizer
import re
def get_tokenizer(tokenizer_path):
    """加载并返回Hugging Face的tokenizer对象"""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path)

# 默认全局tokenizer
_tokenizer_path = "/mnt/nvme/models/Qwen3-0.6B"
tokenizer = get_tokenizer(_tokenizer_path)

def get_cpu_socket_count():
    """获取物理CPU插槽数量（仅Linux）"""
    sockets = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    sockets.add(line.strip().split(":")[1])
        return len(sockets) if sockets else 1
    except Exception:
        return 1
def get_all_jsonl_files(root_dir: str):
    """递归获取所有jsonl文件路径"""
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.jsonl'):
                files.append(os.path.join(dirpath, fname))
    return files

def split_sentences(text):
    """
    按中英文标点符号断句
    支持：中文 。？！，； 英文 .?!,;
    参数:
        text: 输入文本字符串
    返回:
        句子列表
    """
    # 正则表达式匹配中英文句末标点
    pattern = r'([。,！？；.!?;]\s*|\n\s*)'
    sentences = re.split(pattern, text)
    
    # 将标点和句子重新组合
    result = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        if sentence.strip():
            result.append(sentence.strip())
    
    # 处理最后一个句子
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    return result

def suggest_concurrency(task_type: str):
    """
    根据任务类型（'cpu' 或 'io'），获取服务器CPU核心数和插槽数，给出最优并发数建议。
    :param task_type: 'cpu' 表示CPU密集型，'io' 表示IO密集型
    :return: (physical_cores, logical_cores, sockets, best_concurrency)
    """
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    sockets = get_cpu_socket_count()
    if task_type.lower() == 'cpu':
        # CPU密集型，建议用物理核心数或略低于物理核心数
        best = max(1, physical_cores - 1)
    elif task_type.lower() == 'io':
        # IO密集型，建议用逻辑核心数或更高
        best = logical_cores
    else:
        raise ValueError("task_type must be 'cpu' or 'io'")
    return {
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'sockets': sockets,
        'suggested_concurrency': best
    }

def token_word_length(text):
    """计算文本的 token 长度（不包含特殊标记）"""
    sentence_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(sentence_ids)

def sentence_token_count(sentence: str) -> dict:
    """
    输入一个句子，返回{句子: token数量}的字典
    """
    count = token_word_length(sentence)
    return {sentence: count}


def extract_filename(filepath: str) -> str:
    """
    从文件路径中提取倒数第三层目录名（如 literature_emotion）
    参数:
        filepath (str): 完整的文件路径
    返回:
        str: 倒数第三层目录名
    """
    parts = os.path.normpath(filepath).split(os.sep)
    if len(parts) >= 3:
        return parts[-4]
    else:
        return ""


def split_contents_by_token_limit(all_contents, max_token=4096):
    """
    all_contents: dict, {sent: token_num, ...}
    max_token: int, 最大token数
    return: list of dicts, [{'text': ..., 'token': ...}, ...]
    """
    result = []
    current_text = ""
    current_token = 0

    for sent, token_num in all_contents.items():
        # 如果加上当前句子的token数会超过max_token，则先保存当前拼接内容
        if current_token + token_num > max_token:
            if current_text:  # 避免空字符串
                result.append({'text': current_text.strip(), 'token': current_token})
            # 重置
            current_text = ""
            current_token = 0
        # 拼接当前句子
        current_text += sent
        current_token += token_num

    # 最后剩余的内容也要加入
    if current_text:
        result.append({'text': current_text.strip(), 'token': current_token})

    return result
def save_dict_to_file(data: dict, file_path: str):
    """
    将字典以字符串形式写入到指定文件路径
    """
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    print("CPU密集型建议:")
    print(suggest_concurrency('cpu'))
    print("IO密集型建议:")
    print(suggest_concurrency('io'))
    # # token_word_length 测试
    # test_text = "市场监管总局:有效引导餐饮行业公平竞争"
    # print(f"文本: {test_text}")
    # print(f"token数量: {token_word_length(test_text)}")
    # str="北京天堂超市酒吧疫情引发思考: 复工复产不能一刀切\n北京涉天堂超市酒吧聚集性疫情持续发酵引发思考复工复产不能一刀切北京天堂超市酒吧规模聚集性疫情,让首都的防控形势再次严峻复杂.6月13日,北京市疫情防控工作新闻发布会通报,6月9日0时至13日15时,涉天堂超市酒吧聚集性疫情累计报告228例感染者.市疾控中心副主任刘晓峰表示,涉天堂超市酒吧聚集性疫情处于发展阶段,已涉及14个区和经开区的100个街乡,续发病例涉及多个公共场所,呈现点多面广的特点,疫情仍存在传播扩散风险.酒吧为何成为疫情\"放大器\"?复工复产如何避免\"一刀切\"?如何加强对空乘等重点行业从业人员的防疫管理?记者采访了北京大学政府管理学院教授,城市治理研究院执行院长沈体雁,中国人民大学应用经济学院区域与城市经济研究所教授姚永玲,中国人民大学公共管理学院教授马亮.酒吧成为疫情\"放大器\"暴露出哪些防疫短板记者:连日来,天堂超市酒吧规模聚集性疫情成为公众关注的焦点.酒吧为何容易成为疫情\"放大器\"?这暴露出哪些防疫短板?马亮:酒吧是代表城市文化的重要公共场所之一,以夜间经营为主,加之客源广泛而庞杂,人员密集聚集和高度密接,通风条件差,容易成为疫情暴发的关键节点.一些出入酒吧的消费者希望身份隐秘,酒吧为了吸引客源也在防疫措施方面睁一只眼闭一只眼.从通报看,北京天堂超市酒吧存在未按规定扫码,测温,查验核酸检测阴性证明的情况,疫情防控风险集聚.沈体雁:酒吧等休闲娱乐场所是现代城市的重要公共空间,也是除居住空间,工作空间之外的第三空间.相较第一空间和第二空间,第三空间具有很强的流动性,交互性,\"潜水性\"和模糊性.在没有疫情时,第三空间有利于提升城市活力,激发城市创新,彰显城市文化,促进城市经济发展;在疫情防控等应急状态下,第三空间给城市治理提出了全新挑战.必须加强对超大城市第三空间治理模式的研究,构建第三空间精细化治理体系.记者:从统筹推进疫情防控和产业转型升级的角度看,如何加强对酒吧等娱乐场所管理,促进相关产业转型升级?马亮:酒吧,KTV等娱乐场所既是人员聚集和疫情扩散的关键节点,也是防疫战线容易失守的薄弱环节之一.尽管这些娱乐场所的经营时段和方式有一定特殊性,但是不能成为例外,更不能成为防疫的失守短板.可以借助疫情防控的契机,加强对娱乐场所的规范化管理,并带动这些产业转型升级.比如,可以强化疫情防控的政策落实情况,视情况确定娱乐场所的运营资格.复工复产不能一刀切,不可能一蹴而就,应分区分级,分类分时予以推进记者:疫情暴发以来,一些地方分区分级,分类分时推进企业生产和复工复产.复工复产为何不能\"一刀切\"?分区分级,分类分时应注意哪些要点?姚永玲:为避免疫情造成经济的短期停摆转向长期抑制,复工复产势在必行.但复工复产不能\"一刀切\",不能一蹴而就.仅就工作场所而言,不同场所之间就存在差别,有人流密集和聚集性强的场所,也有固定人员安静办公的场所,此外还有室外工作场所.在落实差异化防控策略的前提下,遵循产业链规律,依据特定标准,分区分级,分类分时推进复工复产,才是安全有序,稳健高效的.复工复产行业选择标准的设定,应充分体现重大疫情的特殊性,在产业关联标准的基础上,还应遵循保障基本生活需要,保障疫情防控,保障就业稳定和保障发展质量等标准.企业在复工复产过程中,也要关注其防控与经营的双重角色,切实履行防疫责任.马亮:疫情防控要科学,精准和合理,不同地区,时段和企业的情况千差万别,不能\"一刀切\".复工复产要考虑多数企业特别是制造业企业的产业链存在联动性.复工复产的政策制定要坚持一地一策,一时一策,根据防疫态势变化进行动态管理.应根据本地情况制定预案,在相关条件达到时采取相应政策.天堂超市酒吧规模聚集性疫情暴发后,北京宣布对酒吧等地下场所开展大排查,举一反三,抓好整改,对防控措施落实不到位,地下密闭通风不良的场所暂停开放.对重点行业高风险人群,需要落实责任,加强防疫管理教育,增强各个单位,部门之间的相互衔接和彼此协调记者:6月10日,2名国航空乘核酸检测阳性,2人均为天堂超市酒吧聚集性疫情关联人员.空乘是否属于应加强防疫规范管理的重点行业从业人员?为筑牢疫情防控安全防线,如何加强对重点人群防疫管理?马亮:民航作为与公众出行息息相关的行业,其空乘人员属于疫情之下的高风险人群.必须加强对此类重点行业从业人员的防疫管理.首先,要强化从业人员的防疫意识,明确其应承担的防疫责任.其次,有条件的企业和用人单位应对重点人员采取闭环管理或轮班制等措施,确保防疫经营两不误.最后,相关单位应强化责任意识,落实\"四方责任\",坚持履职尽责,失职追责,坚决防止疫情通过机场,航空器传播扩散.姚永玲:这涉及到城市管理中各个部分之间的相互衔接和彼此协调.大城市是一个由各种人群,各个部门组成的复杂而精细的系统,城市越大,系统越精细,牵一发而动全身.在针对重点人群进行防疫管理过程中,部门协同极为关键.一般规律是根据疫情防控过程中的具体责任和风险程度,由各相关部门进行会商,提出大家都接受并能执行的管理措施.以民航空乘人员为例,一般由医学专家根据风险程度,列出风险点所涉及的不同部门和人员;民航部门根据其员工生活习惯,对可能出险的行为进行合理引导;城市政府根据风险人员需求,提供生活服务.当然,在疫情防控中,还有很多其他部门,其他人员是重点人群,关键是要对风险进行预先评估,为城市政府和相关部门提供具体指导意见,让相关部门参与制定管理措施.精细化的城市治理,就是要针对不同事件,不同场所,不同人员,不同部门拿出有区别的管理措施.记者:最近一些地方暴发的疫情,折射出我们在城市治理方面存在哪些短板?城市治理特别是超大特大城市的治理,能从抗击疫情中收获哪些启示?沈体雁:北京天堂超市酒吧规模聚集性疫情,反映我们城市治理体系与能力仍然存在死角与短板.一是对城市治理规律认识有待加强.对酒吧等城市第三空间的运行特点和治理规律认识不够,导致我们对这类区域的超前谋划不够.二是超大城市治理的体制机制还需进一步完善.对于酒吧等新业态,新空间,新问题,公共卫生,市场监管,文化旅游,城市管理,基层治理等城市公共行政管理部门之间出现管理缺失,相互衔接和彼此协调不够.三是治理手段还不够精准.目前疫情防控的主要感知手段是基于手机信令,健康码等社会感知技术,事实上城市里总有人没有手机,不用手机甚至故意误用手机,电子围栏和大数据感知不到的\"潜水\"人群,给超大城市治理手段提出新课题.四是对大量自由职业者,社会闲散人员等组织化程度不高的市民群体的特点,活动规律,公共生活方式和社会治理需求等研究不够深入.管理死角问题,不能在发生问题后再去探讨,而应事先熟悉管理和服务对象,把工作做在日常,预先考虑和设想各种可能发生的情境,这样才能治\"未病\".必须总结经验教训,探索具有中国特色的松紧平衡,有张有弛的超大城市公共空间,公共生活,公共安全,公共秩序治理体系.姚永玲:城市治理的工作就是不断解决城市发生的问题,城市越繁荣,出现的问题就越复杂,城市治理的任务就越艰巨.大城市人口密度高,人员复杂,流动性强,也正是这些特点使得大城市更需要科学的治理方法.精细化管理是防止\"一刀切\"的有效手段.实现精细化管理的核心是贯彻以人为本的管理理念.在疫情防控中如何以人为本加强城市治理,是我们应该吸取的经验教训.推进以人为核心的新型城镇化建设,紧紧围绕人民的需求来规划,建设和治理城市记者:哪些产业可以留在北京?这一直是困扰北京高质量发展的一个难题.疫情对疏解北京非首都功能有何启示?姚永玲:北京市产业疏解和产业升级是针对长期以来形成的大城市病以及提高城市产业竞争力而言的.大城市病是指交通拥堵,环境污染,住房紧张,社会治安以及失业等不利于城市健康和可持续发展的一些问题.任何大城市都会有这些问题,大城市治理的主要任务就是解决这些问题.一些不适合在北京布局的产业疏解是解决大城市病的长期手段,疏解也是相对而言,并非所有人口聚集的场所都要疏解.在疏解产业的过程中仅有\"白菜心\",没有配套产业,形不成产业链和产业体系,高端产业也就成了无源之水.大城市的高密度人口和高流动人口是创新的源泉,离开了这一点,大城市与中小城市没有区别,没有了创新活力,大城市也就不会存在.如何既保持城市的创新活力,又能减少公共卫生事件的发生,这是全世界大城市治理的难题.只有以人为本的管理理念,精细化的管理手段,以及尊重科学的防控措施,才能实现大城市的现代化治理.马亮:北京的城市定位是政治中心,文化中心,科技创新中心和国际交往中心.与文化创意产业,高新技术,国际交流等相关的产业是北京重点发展的核心产业.与之关联度不高的产业,可以通过疏解非首都功能来实现,可以抓住疫情防控的契机推动产业疏解和转移.城市政府可抓住契机,积极推动核心产业的保障和扶持,推动非首都功能疏解.在城市功能疏解方面要处理好政府和市场的角色和相互关系,避免政府\"越俎代庖\"导致的问题.要尊重城市发展的自然规律,政府顺势而为地加以引导.此外,可以更好地处理存量和增量的关系,按照存量认定既定事实,而增量则予以限制或禁入,推动这些期望疏解的行业逐渐萎缩和退出.记者:一段时间以来,人口过度聚集,让北京资源环境矛盾日益凸显.如何准确理解\"以人为核心的新型城镇化\"?为什么既要持续推进新型城镇化,又要防止人口过多往城市尤其是大城市集中?沈体雁:2013年12月,习近平总书记主持召开改革开放以来首次中央城镇化工作会议,首次系统提出了新型城镇化理念,开启了我国城镇化转型发展的新征程.到2021年,我国常住人口城镇化率达64.72%,户籍人口城镇化率提高到46.7%,人民群众的幸福感,获得感,安全感不断增强,走出了一条中国特色新型城镇化道路.\"以人为核心\"是新型城镇化的根本要求.\"以人为核心\"的实质就是要紧紧围绕人民的需求来规划,建设和治理城市.既做好城市生态修复,功能完善,历史文化保护传承,又统筹优化城市人口密度和空间结构,促进大中小城市和小城镇协调发展,让人民宜居宜业,让城市治理以人为本,科学高效.马亮:过去的城镇化过分强调土地,行政意义上的城镇化,而没有真正聚焦人的城镇化.\"以人为核心的新型城镇化\"就是要以满足人对城市的需求来建设和推进城镇化,比如市民对生态环境,居住条件,交通条件,休闲娱乐等方面的需求,以及老年人,儿童,青年等对城市的不同需求.过度拥挤的城市容易导致\"大城市病\",毫无疑问不是\"以人为核心的新型城镇化\".要特别避免城市摊大饼式的盲目扩张带来的问题,导致城市功能无法满足人对城市的多元需求.姚永玲:城镇化不是一个简单的城市人口比例和面积的扩张,而是在社会保障,生活方式等方面实现由乡到城的转变.推进以人为核心的新型城镇化建设,应当通过深化户籍制度等关键领域的改革,推动以基本公共服务均等化为中心内容的农民工市民化.一方面,要推进城乡基本公共服务均等化,继续把新增公共资源向农村倾斜,提高农村居民享受基本公共服务的水平,让进城农民工及其家庭真正融入城市,享受同等的社会保障,义务教育,保障性住房等基本公共服务.另一方面,要着力提高户籍人口城镇化率,通过户籍制度改革实现农民工市民化,这是推进基本公共服务均等化的最有效手段和最终体现.推进以人为核心的新型城镇化建设,必须在更深层次上认识城镇化的规律以及城镇化与其他经济社会发展领域的关系,其关键就在于是否立足于以人民为中心.只有以人民为中心,顺应发展规律的城镇化,才能提高发展的共享性和可持续性."
    # liststr=split_sentences(str)
    # for i in liststr:
    #     print(i)
        # print(sentence_token_count(i))
    # all_contents = {
    #     "Hello. ": 1000,
    #     "How are you? ": 1500,
    #     "I'm fine. ": 1200,
    #     "Thank you! ": 800,
    #     "Bye. ": 500
    # }

    # res = split_contents_by_token_limit(all_contents, max_token=4096)
    # for item in res:
    #     print(item)
    s = "/mnt/nvme/data/process/testdata/literature_emotion/chinese/high/trank_00064.jsonl"
    print(extract_filename(s))  # 输出: literature_emotion